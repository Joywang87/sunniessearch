# Following tutorial at https://hackernoon.com/web-scraping-tutorial-with-python-tips-and-tricks-db070e70e071

from bs4 import BeautifulSoup
import requests
import shutil
import csv

# I got this link by hand when searching on the website. just add the page number at end
base_page_link ='https://www.zappos.com/sunglasses/CKzXARCq2QHiAgIBAg.zso?s=goliveRecentSalesStyle/desc/&p='
MAX_PAGES = 1
all_item_urls = []

filename = 'sunglasses.csv'
headers = ['title','number','style','url', 'price']
with open(filename, 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter = ',')
    csv_writer.writerow(headers)
    for page_number in range(1,MAX_PAGES+1):
        page_link = base_page_link + '{}'.format(page_number)
    
        # fetch the content from url
        page_response = requests.get(page_link, timeout=5)
        # parse html
        page_content = BeautifulSoup(page_response.content, "html.parser")
    
        # These are all the tags on the search page that are links to products
        item_tags = page_content.find_all('a')
        item_urls = [item_tag.get('href') for item_tag in item_tags if (item_tag.get('itemprop') == 'url' or item_tag.get('itemProp') == 'url')]
        all_item_urls.extend(item_urls)
            
        base_filename = 'images/'
        # I got the tag names from just looking at the source of the product pages
        for url in item_urls:
            try:
                full_url = 'https://www.zappos.com{}'.format(url)
                print('\nDownloading {}'.format(full_url))
                page_response = requests.get(full_url, timeout=5)
                page_content = BeautifulSoup(page_response.content, "html.parser")
            
                product_number = page_content.find('meta', attrs={'name' : 'branch:deeplink:product'}).get('content') 
                product_style = page_content.find('meta', attrs={'name' : 'branch:deeplink:style'}).get('content')
                product_title = page_content.find('meta', attrs={'name' : 'og:title'}).get('content')
                product_price = page_content.find('meta', attrs={'name' : 'branch:deeplink:style'}).get('content')
                product_url = page_content.find('meta', attrs={'name' : 'og:url'}).get('content') 
                 
                print(product_title)
                print(product_url)
                image_tags = page_content.find_all('img')
                image_urls = [image_tag.get('src') for image_tag in image_tags if (image_tag.get('alt') == 'Product View')]
                for index, image_url in enumerate(image_urls):
                    output_filename = base_filename + '{}_{}_{}.jpg'.format(product_number, product_style, index) 
                    response = requests.get(image_url, stream=True)
                    with open(output_filename, 'wb') as out_file:
                        shutil.copyfileobj(response.raw, out_file)
                    del response
                row = [product_title, product_number, product_style, product_url,product_price]
                csv_writer.writerow(row)
                csvfile.flush()
            except:
                print('Error')