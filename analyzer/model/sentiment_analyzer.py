from fake_useragent import UserAgent
import requests
from bs4 import BeautifulSoup
import math
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import re
import pandas as pd
import random
import string
from tqdm import tqdm
from analyzer.apps import AnalyzerConfig

class Sentiment_analysis:
    proxies=None
    reviews_dict=None
    url=None
    max_try=10
    resultant=-1
    total_try=5
    def __init__(self,url):
        self.url=url
        self.reviews_dict = { "content":[]}
        self.proxies = self.proxy_generator()
        self.start_page = 1 
        self.end_page = self.total_pages()
        if(self.end_page>15):
            self.total_try=self.max_try=3
        for i in tqdm(range(self.start_page, self.end_page+1)):
            self.page_scraper(i)
        self.reviews_df=pd.DataFrame(self.reviews_dict)
        self.reviews_df["content"]=self.reviews_df["content"].apply(self.data_cleaner)
        self.prediction()

    def total_pages(self):
        response = self.request_wrapper(self.url.format(1))
        soup = BeautifulSoup(response.text, 'html.parser')     
        content = soup.find_all("div", {"data-hook": "cr-filter-info-review-rating-count"})
        val=content[0].get_text().translate(str.maketrans('', '', string.punctuation)).strip().split(" ")
        if val[3]=='':
            total_reviews=int(val[4])
        else:
            total_reviews=int(val[3])
        total_pages = math.ceil(total_reviews/10)
        return total_pages

    def request_wrapper(self, url):
        self.proxy = random.choice(self.proxies)
        self.user=UserAgent().random
        while True:
            response = requests.get(url, verify=False, headers={'User-Agent':self.user},proxies=self.proxy)
            if (response.status_code !=200):
                raise Exception("Url is not valid")
            if "api-services-support@amazon.com" in response.text:
                if self.max_try==0:
                    raise Exception("CAPTCHA is not bypassed")
                else:
                    self.max_try-=1
                    self.user=UserAgent().random
                    self.proxy = random.choice(self.proxies)
                    continue
            self.max_try=self.total_try
            break
        return response

    def page_scraper(self, page):
        try:
            response = self.request_wrapper(self.url.format(page))   
            soup = BeautifulSoup(response.text, 'html.parser')
            reviews = soup.findAll("div", {"class":"a-section review aok-relative"})
            reviews = BeautifulSoup('<br/>'.join([str(tag) for tag in reviews]), 'html.parser')
            contents = reviews.find_all("span", {"data-hook":"review-body"})
            content_lst = []
            for content in contents:
                text_ = content.get_text().replace("\n", " ").strip()
                content_lst.append(text_) 
            self.reviews_dict['content'].extend(content_lst)
        except Exception as e:
            print ("Not able to scrape page {}".format(page), flush=True)
    def proxy_generator(self):
        response=requests.get("https://api.proxyscrape.com/?request=displayproxies&protocol=socks5&timeout=10000&country=all")
        soup = BeautifulSoup(response.content, 'html.parser')
        proxy_data = soup.findAll(text=True)[0].split("\r\n")
        proxies = [{'http':'http://'+proxy} for proxy in proxy_data]
        return proxies

    def data_cleaner(self, String):
        stopWord=stopwords.words("english")
        stopWord.remove('not')
        tags = re.compile('<.*?>')
        string_without_tags = re.sub(tags, " ", String)
        string_without_url = re.sub(r'[\S]*\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?|[\S]*@gmail','',string_without_tags)
        string_without_punc=re.sub(r'[^a-zA-Z]',' ',string_without_url)
        noise_free_review=[i for i in string_without_punc.split() if i not in stopWord ]
        noise_free_review=' '.join(noise_free_review)
        return noise_free_review

    def prediction(self):
        reviews_array=self.reviews_df['content'].values
        x_data=AnalyzerConfig.tokenizer.texts_to_sequences(reviews_array)
        x_data=pad_sequences(x_data,padding="pre")
        ans=AnalyzerConfig.loaded_model.predict(x_data)
        self.poscount=0
        self.negcount=0
        for i in ans:
            if(i>=0.5):
                self.poscount+=1
            else:
                self.negcount+=1
        checkvalue=self.poscount*(0.5)
        if(checkvalue>self.negcount):
            self.resultant= "product is worth to buy"
        else:
            self.resultant="product is not worth to buy"


