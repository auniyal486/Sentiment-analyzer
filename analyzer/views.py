from .model.sentiment_analyzer import Sentiment_analysis
import requests
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
from urllib.parse import urlparse, unquote

class analyzing_reviews(APIView):
    def get(self,request):
        if request.GET.get("url", None) is not None:
            url = unquote(request.GET["url"])
            try:
                domain_name = urlparse(request.GET["url"]).netloc
            except:
                domain_name = None
            if domain_name!= "www.amazon.in":
                response={'response':"Please provide correct product url"}  
                return Response(response,status=status.HTTP_422_UNPROCESSABLE_ENTITY)
            try:
                url_parts = url.split("/")
                dp_index = url_parts.index("dp")
                asin_number = url_parts[dp_index+1]
            except:
                response={'response':"Please provide correct product url"}  
                return Response(response,status=status.HTTP_422_UNPROCESSABLE_ENTITY)
            url="https://"+domain_name+ "/product-reviews/" + asin_number + "?pageNumber={}"
            r = requests.get(url,timeout=0.5,verify=False)
            if not (r.status_code == 200 or r.status_code == 503):
                response={'response':"Please provide correct product url"}  
                return Response(response,status=status.HTTP_422_UNPROCESSABLE_ENTITY)
            print(url)
            try:
                task=Sentiment_analysis(url)
            except Exception as e:
                print(e)
                response={'response':'Something is wrong... please try after some time'}  
                return Response(response,status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            response={'result':task.resultant,'positives':task.positive,'negatives':task.negative,'netural':task.netural}
            return Response(response)
        else:
            response={'response':"Please provide product url"}  
            return Response(response,status=status.HTTP_400_BAD_REQUEST)

class Home(APIView):  
    def get(self, request):  
        return Response("Reviews Scraper And Analyzer")  