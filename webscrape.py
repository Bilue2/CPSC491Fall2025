from bs4 import BeautifulSoup
import requests 

html_text = requests.get("https://www.iafc.org/blogs/blog/iafc/2024/06/28/federal-communications-commission-adopts-rule-requiring-location-based-routing-for-9-1-1-calls/").text 

#print(html_text) 

soup = BeautifulSoup(html_text, 'lxml')
# print(soup.prettify())
headers = soup.find_all('h1') 
paragraphs = soup.find_all('p')

for header in headers: 
    title = header.text 
    print(title) 

for p in paragraphs:
    content = p.text
    print(content) 


