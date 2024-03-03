import os
import sys
import multiprocessing
import six

from urllib.request import urlopen, urlretrieve

def _worker(url_and_target):
    """child processor"""
    try:
        (url, target_path) = url_and_target
        print(">>>Downloading  " + target_path)
        urlretrieve(url, target_path) # download zip files
    except (KeyboardInterrupt, SystemExit):
        print('Existing child process')

class KGSIndex:
    def __init__(self,
                 kgs_url='https://u-go.net/gamerecords/',
                 index_page='kgs_index.html',
                 data_dir='data/raw'):
        self.kgs_url = kgs_url
        self.index_page = index_page
        self.data_dir = os.getcwd() + '/go/' + data_dir
        self.file_infos = []
        self.urls = []
        self._load_index()

    def _load_index(self):
        """Create the actual index representation from the previously downloaded or cached html."""
        index_contents = self._create_index_page() # get the index content from html file
        # split the whole string and select the string that contain "https://"
        split_page = [item for item in index_contents.split('<a href="') if item.startswith("https://")]
        for item in split_page:
            download_url = item.split('">Download')[0] # split the string again into two and pick the first string that contain url "https://"

            if download_url.endswith('.tar.gz'): # only need the url that end with .tar.gz
                self.urls.append(download_url)
        for url in self.urls:
            filename = os.path.basename(url) # get basename from url
            split_file_name = filename.split('-') # sprint into 5 strings
            num_games = int(split_file_name[len(split_file_name) - 2]) # the num games at index 3
            print(filename + ' ' + str(num_games))
            # store file infos
            self.file_infos.append({'url': url, 'filename': filename, 'num_games': num_games})
    
    def _create_index_page(self):
        """If there is no local html contain links to files, create one."""
        if os.path.isfile(self.index_page): # look for html index page in working directory
            print('Reading cached index page')
            index_file = open(self.index_page, 'r') # open the html file
            index_contents = index_file.read() # read it
            index_file.close()
        else:
            print("Downloading index page")
            fp = urlopen(self.kgs_url) # open the url to get the html index page
            index_contents = six.text_type(fp.read()) # convert binary to text
            fp.close()
            index_file = open(self.index_page, 'w') # write the index content into the html file
            index_file.write(index_contents) 
            index_file.close()
        return index_contents
    
    def download_files(self):
        """Download zip files"""
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir) # create the folder if not exist
        
        urls_to_download = [] # contain urls and file_paths to download
        for file_info in self.file_infos:
            url = file_info['url']
            file_name = file_info['filename']
            file_path = self.data_dir + '/' + file_name
            if not os.path.isfile(file_path):
                urls_to_download.append((url, file_path))
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores) # create child processors
        try:
            it = pool.imap(_worker, urls_to_download) # multiprocessing
            for _ in it:
                pass
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            pool.terminate()
            pool.join()
            sys.exit(-1)


if __name__ == '__main__':
    index = KGSIndex()
    index.download_files()