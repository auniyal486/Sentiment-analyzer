from .model.sentiment_analyzer import Sentiment_analysis
import requests
from rest_framework.views import APIView
from django.http import JsonResponse
import urllib.parse

class analyzing_reviews(APIView):
    def get(self,request):
        if request.GET.get("url", None) is not None:
            url=urllib.parse.unquote(request.GET["url"])
            domain=url.lower().find('amazon')
            if(domain==-1):
                response={'response':-1}
                return JsonResponse(response)
            url_parts=url[domain:].split("/")
            url=url_parts[0]
            if(url_parts[1]=='dp'):
                asin_number=url_parts[2]
            else:
                asin_number=url_parts[3]
            url="https://www." +url+ "/product-reviews/" + asin_number + "?pageNumber={}"
            if(requests.get(url)==False):
                response={'response':-1}
                return JsonResponse(response)
            try:
                task=Sentiment_analysis(url)
            except Exception as e:
                response={'response':0}  
                return JsonResponse(response)
            response={'response':task.resultant,'positives':task.poscount,'negatives':task.negcount}
            return JsonResponse(response)
        else:
            response={'response':-1}  
            return JsonResponse(response)