from bs4 import BeautifulSoup
import time
import requests
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
from collections import defaultdict, Counter, namedtuple
from os.path import isfile, isdir
import os
from nltk import word_tokenize
from nltk.stem.snowball import ItalianStemmer
from nltk.corpus import stopwords
import nltk
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import heapdict
from scipy import sparse
from wordcloud import WordCloud


def timeit(method):
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        print('%r  %2.2f s' % (method.__name__, (te - ts)))
        return result
    return timed


class AdCluster:
    def __init__(self):
        self.data_dir = "./data/"
        if not isdir(self.data_dir):
            os.mkdir(self.data_dir)

        self.info = None
        self.desc = None
        self.desc_index = None

        self.stemmer = ItalianStemmer()
        self.stop_words = set(stopwords.words('italian'))

        self.vocab = None
        self.documents = None
        self.inv_index = None  # no inverted index yet available
        self.idf = None  # no inverse document frequency yet available

        self.nltk_check_downloaded()

        self.url_base = "https://www.immobiliare.it"
        self.url_search = "/vendita-case/roma/?criterio=rilevanza&pag="
        try:
            html = requests.get(self.url_base + self.url_search + "1").content
            soup = BeautifulSoup(html, "html.parser")

            pag_number_list = soup.find("ul", class_="pagination pagination__number")
            self.max_pag_nr = int(pag_number_list.find_all("li")[-1].text)
        except requests.exceptions.ConnectionError:
            pass

    def load_data(self, info_fname, desc_fname, convert_to_tfidf=True, skip_scrape=False):
        info_file = self.data_dir + info_fname
        desc_file = self.data_dir + desc_fname
        info_exists = isfile(info_file)
        desc_exists = isfile(desc_file)
        if info_exists and desc_exists:
            info, desc = pd.read_csv(info_file, sep=",", index_col=None, header=None), \
                         pd.read_csv(desc_file, sep=",", index_col=None, header=None)
        elif not skip_scrape:
            info, desc = self.scrape_immobiliare()
        else:
            raise ValueError(f"No files present and 'skip_scrape'={skip_scrape}.")

        info.drop(columns=[0], inplace=True)
        info.columns = ['ID', 'Price', 'Rooms', 'Area', 'Bathrooms', 'Floor']
        desc.drop(columns=[0, 2], inplace=True)
        desc.columns = ['ID', 'Description']

        info.reset_index(drop=True, inplace=True)
        desc.reset_index(drop=True, inplace=True)
        info[info["Floor"] == "A"] = 12
        info[info["Floor"].isin(("R", "T"))] = 0
        info[info["Floor"] == "S"] = -1

        # remove duplicates
        info = info.loc[(-1 * info["ID"].duplicated(keep=False) + 1).astype(bool)]
        desc = desc.loc[(-1 * desc["ID"].duplicated(keep=False) + 1).astype(bool)]

        desc_ids = desc["ID"]
        info_ids = info["ID"]
        info_corr = info_ids[info_ids.isin(desc_ids)]
        desc_corr = desc_ids[desc_ids.isin(info_ids)]
        rem_ids = pd.unique(pd.concat((info_corr, desc_corr)))

        info = info[info["ID"].isin(rem_ids)]
        desc = desc[desc["ID"].isin(rem_ids)]

        nans = lambda df: df.isnull().any(axis=1)  # handy func to find NANs

        nan_info = nans(info)
        nan_desc = nans(desc)
        # drop all ads where any of the two matrices encounter NANs
        info = info.drop(index=info.index[nan_info | nan_desc]).reset_index(drop=True)
        desc = desc.drop(index=desc.index[nan_info | nan_desc]).reset_index(drop=True)

        if convert_to_tfidf:
            desc = self.build_desc_matrix(desc)

        self.info = info
        self.desc = desc

        return info, desc

    @staticmethod
    def get_ad_from_url(url, parser):
        response = requests.get(url)

        html_soup = BeautifulSoup(response.text, parser)
        ad_containers = html_soup.find_all('p', class_='titolo text-primary')

        urls = []

        for container in ad_containers:
            if "/nuove_costruzioni/" not in container.a['href']:
                urls.append(container.a['href'])

        return urls

    @staticmethod
    def get_data(url):

        id = re.findall(r'(\d+)', url)[0]  # Get ad ID parsing the url

        response = requests.get(url)

        html_soup = BeautifulSoup(response.text, 'html.parser')
        data_container = html_soup.find('ul', class_='list-inline list-piped features__list')

        if data_container is not None:
            find = lambda itm: itm.find('div', class_='features__label')

            for item in data_container.children:

                # Locate rooms number
                found = find(item)
                if found:
                    if found.contents[0] == 'locali':
                        rooms = item.find('span', class_='text-bold').contents[0]
                        rooms = re.sub('[^A-Za-z0-9]+', '', rooms)

                    # Locate surface extension
                    elif found.contents[0] == 'superficie':
                        area = item.find('span', class_='text-bold').contents[0]
                        area = re.sub('[^A-Za-z0-9]+', '', area)

                    # Locate bathrooms number
                    elif found.contents[0] == 'bagni':
                        bathrooms = item.find('span', class_='text-bold').contents[0]
                        bathrooms = re.sub('[^A-Za-z0-9]+', '', bathrooms)

                    # Locate floor number
                    elif found.contents[0] == 'piano':
                        floor = item.find('abbr', class_='text-bold').contents[0]
                        floor = re.sub('[^A-Za-z0-9]+', '', floor)

                # Extract the description
                try:
                    cl = 'col-xs-12 description-text text-compressed'
                    description = html_soup.find('div', class_=cl).div.contents[0]
                    description = re.sub('[^a-zA-Z0-9-_*. ]', '', description)  # Remove special characters
                    description = description.lstrip(' ')  # Remove leading blank spaces
                except AttributeError:
                    return False

        try:
            return [[id, rooms, area, bathrooms, floor], [id, description]]
        except NameError:
            return False

    def scrape_immobiliare(self):
        row_info, row_desc, url_list = [], [], []

        try:
            import lxml
            parser = "lxml"
        except ImportError:
            parser = "html.parser"

        base_url = "https://www.immobiliare.it/vendita-case/roma/?criterio=rilevanza&pag="

        for i in tqdm(range(450)):
            url_list += self.get_ad_from_url(base_url + str(i), parser)

        for url in tqdm(url_list):

            print(url)

            # This while loop is needed to retry the request in case of connection error
            while True:
                try:
                    cont = self.get_data(url)
                    if cont:
                        # Convert list in dataframe
                        row_data = np.asarray(cont[0]).reshape(1, 5)
                        row_data = pd.DataFrame(data=row_data,
                                                columns=['ID', 'Rooms', 'Area', 'Bathrooms', 'Floor'])

                        # Append results to info dataframe
                        row_info.append(row_data)

                        # Convert list in dataframe
                        row_description = pd.np.asarray(cont[1]).reshape(1, 2)
                        row_description = pd.DataFrame(data=row_description,
                                                       columns=['ID', 'Description'])

                        # Append results to description dataframe
                        row_desc.append(row_description)

                        # Create two csv files line by line
                        with open('data/data.csv', 'a') as f:
                            row_data.to_csv(f, header=False)
                        with open('data/description.csv', 'a') as f:
                            row_description.to_csv(f, header=False)

                # Wait a second in case of connection error and retry
                except ConnectionError:
                    print('Connection Error')
                    time.sleep(1)
                    continue
                break

        info = pd.concat(row_info)
        desc = pd.concat(row_desc)

        # remove duplicates
        info = info.loc[(-1 * info["ID"].duplicated(keep=False) + 1).astype(bool)]
        desc = desc.loc[(-1 * desc["ID"].duplicated(keep=False) + 1).astype(bool)]

        desc_ids = desc["ID"]
        info_ids = info["ID"]
        info_corr = info_ids[info_ids.isin(desc_ids)]
        desc_corr = desc_ids[desc_ids.isin(info_ids)]
        rem_ids = pd.unique(pd.concat((info_corr, desc_corr)))

        info = info[info["ID"].isin(rem_ids)]
        desc = desc[desc["ID"].isin(rem_ids)]

        nans = lambda df: df.isnull().any(axis=1)  # handy func to find NANs

        nan_info = nans(info)
        nan_desc = nans(desc)
        # drop all ads where any of the two matrices encounter NANs
        info = info.drop(index=info.index[nan_info | nan_desc])
        desc = desc.drop(index=desc.index[nan_info | nan_desc])

        return info, desc

    @timeit
    def build_desc_matrix(self, desc_df):
        self.desc_index = desc_df.index
        docs = desc_df["Description"]
        docs = self._process_docs(docs)
        self._build_invert_idx(docs, proc=False)

        # In the following, the one-hot-encoding of the relevant documents is computed
        # and its tfidf values stored in sparse matrix.
        col = []  # list of non zero column indices
        row = []  # list of non zero row indices
        data = []  # data of the non zero indices
        for d_nr, content in docs.items():
            for term in content:
                col.append(self.vocab.loc[term, "term_id"])
                row.append(d_nr)
                # find the tfidf (the data) of the term in this document
                found = False
                for termset in self.inv_index[term]:
                    if termset.docID == d_nr:
                        data.append(termset.tfidf)
                        found = True
                        break  # value found, no other termset needs to be found after
                if not found:
                    raise ValueError(f"Term {term} in document {d_nr} not found.")
        shape = len(docs), len(self.vocab)
        desc_sparse = sparse.csr_matrix((data, (row, col)), shape=shape, dtype=float)
        return desc_sparse

    def desc_sparse_to_dense(self, desc_sparse):
        if isinstance(desc_sparse, sparse.csr_matrix):
            return pd.DataFrame(desc_sparse.toarray(),
                                index=self.desc_index,
                                columns=self.vocab.index)
        else:
            return desc_sparse

    @staticmethod
    def cluster_kmeans_elbow(X, normalize_=False):
        if normalize_:
            X_clust = normalize(X)
        else:
            X_clust = X
        i = 0
        ks, fits, scores = [], [], []
        while True:
            new_range = [k for k in range(10 * i + 1, 10 * i + 11)]
            ks += new_range
            KM = [KMeans(n_clusters=i) for i in new_range]
            f = [km.fit(X_clust) for km in KM]
            fits += f
            scores += [km.inertia_ for km in f]
            plt.plot(ks, scores)
            plt.show()
            print("Choose number of clusters: ", end="")
            new_k = input()
            if new_k != "":
                try:
                    new_k = int(new_k)
                    if new_k > 0:
                        break
                except ValueError:
                    pass
            i += 1
        km_fit = fits[new_k-1]
        return km_fit

    def find_similar_clusters(self, clusters_info, clusters_desc):
        if self.info is None or self.desc is None:
            raise ValueError("Information and/or description dataframe not yet assigned.")

        labels_info = clusters_info.predict(self.info)
        labels_desc = clusters_desc.predict(self.desc)
        n_clusters_info = clusters_info.n_clusters
        n_clusters_desc = clusters_desc.n_clusters

        cluster_sim = heapdict.heapdict()
        for i in range(n_clusters_info):
            ind_info = np.where(labels_info == i)[0]
            for j in range(n_clusters_desc):
                ind_desc = np.where(labels_desc == j)[0]
                all_ind = np.concatenate((ind_info, ind_desc))
                intersec = 0
                if len(ind_info) < len(ind_desc):
                    for idx in ind_info:
                        if idx in ind_desc:
                            intersec += 1
                else:
                    for idx in ind_desc:
                        if idx in ind_info:
                            intersec += 1
                union = np.unique(all_ind)
                cluster_sim[i, j] = -intersec / len(union)

        return cluster_sim

    def top_words_clusters(self, data, labels, nr_top_k_words):
        top_data = data.apply(
            lambda x: pd.Series(x.sort_values(ascending=False).iloc[:nr_top_k_words].index,
                                index=[f"top{i}" for i in range(1, nr_top_k_words + 1)]),
            axis=1
        )
        _, desc_df = self.load_data("data.csv", "description.csv", convert_to_tfidf=False)
        desc_df = self._process_docs(desc_df["Description"], stem=False)

        top_data["cluster"] = labels
        top_data.sort_values(by=["cluster"], inplace=True)

        for cluster in pd.unique(top_data["cluster"]):
            d = top_data[top_data["cluster"] == cluster].drop(columns=["cluster"])
            freqs = dict()
            for x in d.itertuples():
                idx = x.Index
                actual_ad = desc_df[idx]
                words = x[1:]
                for word in words:
                    for act_w in actual_ad:
                        if self.stemmer.stem(act_w) == word:
                            actual_word = act_w
                            break
                    freqs[actual_word] = data.loc[idx, word]

            wordcloud = WordCloud(width=1600, height=800, background_color="white")
            wordcloud.generate_from_frequencies(freqs)
            plt.figure(num=None, figsize=(20, 10), facecolor='w', edgecolor='k')
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Cluster {cluster} word-cloud of top {nr_top_k_words} words of each ad within cluster.\n"
                      f"The size of words corresponds to their TFIDF value.")
            plt.show()

        return top_data

    @timeit
    def _create_vocab(self, docs, proc=True):
        """
        Creates the vocabulary from documents or reads the vocabulary from file.
        The name is always "vocabulary.csv" containing the word as index and its
        term id as column entry.
        :param docs: dict or pd.DataFrame, the collection of documents (only essential parts)
        :return: the vocabulary
        """

        fname = f"{self.data_dir}vocabulary.csv"
        if proc:
            docs = self._process_docs(docs)
        self.vocab = set()
        for doc in docs.values():
            self.vocab.update(doc)
        self.vocab = pd.DataFrame(pd.Series(np.arange(len(self.vocab)), index=self.vocab),
                                  columns=["term_id"])
        self.vocab.to_csv(fname)
        return self.vocab

    def _process_text(self, text, stem=True):
        """
        Remove special characters and superfluous whitespaces from text body. Also
        send text to lower case, tokenize and stem the terms.
        :param text: str, the text to process.
        :return: generator, yields the processed words in iteration
        """
        if stem:
            stem_func = self.stemmer.stem
        else:
            stem_func = lambda x: x

        text = self.doc_to_string(text).lower()
        sub_re = r"[^A-Za-z']"
        text = re.sub(sub_re, " ", text)
        for i in word_tokenize(text):
            if i not in self.stop_words:
                w = stem_func(i)
                if len(w) > 1:
                    yield(w)

    def _process_docs(self, docs=None, stem=True):
        """
        Takes a collection of documents and processes them iteratively. The docs
        can be a pd.DataFrame, pd.Series or dictionary.
        :param docs: pd.DataFrame, pd.Series or dictionary
        :return: dict, indexed by doc number and lists (processed doc) as values
        """

        if isinstance(docs, pd.DataFrame):
            docs_generator = docs.iterrows()
        elif isinstance(docs, pd.Series):
            docs_generator = docs.iteritems()
        elif isinstance(docs, dict):
            docs_generator = docs.items()
        else:
            raise ValueError("Container type has no handler.")

        d_out = dict()
        for docnr, doc in docs_generator:
            d_out[docnr] = list(self._process_text(doc, stem=stem))
        return d_out

    @staticmethod
    def nltk_check_downloaded():
        """
        Check the prerequisite NLTK tools, download if not found
        """

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    @staticmethod
    def doc_to_string(doc):
        """
        Converts a document to a string. Can take a list, DataFrame, tuple to convert to str
        :param doc: iterable, container of the document
        :return: str, the to string converted document.
        """

        if isinstance(doc, str):
            return doc
        elif isinstance(doc, np.ndarray):
            doc = " ".join(list(map(str, doc.flatten())))
        elif isinstance(doc, (list, tuple)):
            doc = " ".join(doc)
        elif isinstance(doc, (pd.DataFrame, pd.Series)):
            doc = " ".join(list(map(str, doc.values.flatten())))
        else:
            raise ValueError(f"Can't convert file type {type(doc)} to string.")
        return doc

    @timeit
    def _build_invert_idx(self, docs=None, proc=False, read_fname="inverted_index.txt",
                          write_fname="inverted_index.txt", load_from_file=False):
        """
        Build the inverted index for the terms in a collection of documents. Will load a
        previously build inverted index from file if it detects the file existing (and
        param load_from_file is True).
        :param docs: pd.DataFrame/dict, collection of documents
        :param read_fname: str, filename of the inverted txt to load. Needs to be built in the
                                specified way of the method
        :param write_fname: str, filename to write the inverted index to.
        :param load_from_file: bool, load the index from the filename provided if True
        :return: dict, the inverted index with terms as keys and [TermSet(docID, tfidf),...]
                       as values.
        """

        if self.vocab is None:
            self._create_vocab(docs, proc=proc)
        file = f"{self.data_dir}{read_fname}"
        TermSet = namedtuple("TermSet", "docID tfidf")
        if isfile(file) and load_from_file:
            idf_dict = dict()
            inv_index = dict()
            with open(file, "r") as f:
                # load all the information from the file into memory
                for rowidx, line in enumerate(f):
                    if rowidx > 0:
                        term, idf_doclist = line.strip().split(":", 1)
                        idf, doclist = idf_doclist.split("|", 1)
                        idf_dict[term] = idf
                        doclist = list(map(lambda x: re.search(r"\d+,\s?(\d[.])?\d+", x).group().split(","),
                                           doclist.split(";")))
                        inv_index[term] = [TermSet(*list(map(float, docl))) for docl in doclist]
        else:
            # the final inverted index container, defaultdict, so that new terms
            # can be searched and get an empty list back
            inv_index = defaultdict(list)
            docs, idf_dict, term_freqs, doc_counters = self._build_idf(docs, proc)
            for docnr, doc in docs.items():
                # weird, frequency pairs for this document
                freqs = doc_counters[docnr]
                for word, word_freq in freqs.items():
                    # nr of words in this document
                    n_terms = sum(freqs.values())
                    # store which document and frequency
                    inv_index[word].append(TermSet(docnr, word_freq / n_terms * idf_dict[word]))
            # write the built index to file
            with open(f"{self.data_dir}{write_fname}", "w") as f:
                f.write("Word: [Documents list]\n")
                for word, docs in inv_index.items():
                    docs = [(doc.docID, doc.tfidf) for doc in docs]
                    f.write(f"{word}: {idf_dict[word]} | {';'.join([str(doc) for doc in docs])}\n")
        self.inv_index = inv_index
        self.idf = idf_dict
        return inv_index

    @timeit
    def _build_idf(self, docs, proc=True):
        """
        Builds the IDF values for terms in docs.
        :param docs: dict/pd.DataFrame, the documents
        :return: tuple; a tuple of (docs_dict, idf_dict, termFrequencies_dict, docCounters_dict).
        The idf_dict contains the IDF value for each term in the documents.
        The termFrequencies_dict contains the global number of occurences of each term in all the docs.
        the docCounters_dict contains the local number of occurences of each term in the respective doc.
        """

        if proc:
            docs = self._process_docs(docs)
        nr_docs = len(docs)
        idf = defaultdict(lambda: np.math.log(len(docs) + 1))
        # dict to track nr of occurences of each term
        term_freqs = dict()
        # dict to store counter of words in each doc
        doc_counters = dict()
        for docnr, doc in docs.items():
            freqs = Counter(doc)
            doc_counters[docnr] = freqs
            for word in freqs.keys():
                if word in term_freqs:
                    term_freqs[word] += 1
                else:
                    term_freqs[word] = 1
        for word in self.vocab.index:
            # nr of documents with this term in it
            nr_d_with_term = term_freqs[word]
            # inverse document frequency for this term and this document
            idf[word] = np.math.log((float(nr_docs + 1) / (1 + nr_d_with_term)))
        self.idf = idf
        return docs, idf, term_freqs, doc_counters


if __name__ == '__main__':

    clusterer = AdCluster()
    # load data first
    info, desc = clusterer.load_data("data.csv", "description.csv", skip_scrape=True)
    # cluster both tables
    info = info.loc[0:, :]
    desc = clusterer.desc_sparse_to_dense(desc).loc[0:, :]
    info_cluster = clusterer.cluster_kmeans_elbow(info)
    desc_cluster = clusterer.cluster_kmeans_elbow(desc, normalize_=True)
    # check similar clusters
    clusters_sim = clusterer.find_similar_clusters(info_cluster, desc_cluster)
    for i in range(3):
        (clust_info, clust_desc), sim = clusters_sim.popitem()
        sim = -sim
        print(f"Data-Cluster: {clust_info}, Description-Cluster: {clust_desc}, Similarity: {100*sim:.1f}%")

    clusterer.top_words_clusters(clusterer.desc_sparse_to_dense(desc), desc_cluster.labels_, 3)
