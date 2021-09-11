import requests,aiohttp,asyncio
import json,time
from lxml import etree
import os
import ssl
print(ssl.HAS_SNI)

class DateUrl():
    def __init__(self):
        self.url = 'https://chroniclingamerica.loc.gov/data/batches/au_abernethy_ver01/data/sn85044812/'
        self.headers = {
            'cookie': '__cfduid=d870b32a7a66cee05d738eef09aee5e1e1616133298; cf_chl_2=58cc7e01f1ce4bd; cf_chl_prog=a19; cf_clearance=f450c2d630d6842d18cf7f22e3d9163ada06ae2c-1616154124-0-250',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36'
        }
        self.count = 359
        self.file1 = open('demo.txt','w',encoding='utf-8')

    # pages
    def get_page(self):
        pages_text = requests.get(self.url,headers=self.headers).text
        return pages_text

    # parse pages data in corresponding url
    def page_data_parse(self, pages_text):
        html = etree.HTML(pages_text)
        page_link_list = html.xpath('/html/body/table/tbody/tr[position()>1]/td[1]/a/@href')
        page3_urls = []
        for index,link in enumerate(page_link_list):
            page2_link = self.url+link
            page3_text = requests.get(page2_link, headers=self.headers).text
            page3_html = etree.HTML(page3_text)
            page3_link_list = page3_html.xpath('/html/body/table/tbody/tr[position()>1 and position()<last()-12]/td[1]/a/@href')
            print(f'page3_link_list:{len(page3_link_list)}')
            for url in page3_link_list:
                page3_url = page2_link + url
                page3_urls.append(page3_url)
                self.file1.write(page3_url + '\n')
            print(f'finish collecting {index+1} th file path')
        return page3_urls

    # save data
    async def save_data(self, detail_url, session):
        # get detailed info of data
        response_text = requests.get(detail_url, headers=self.headers).text
        html = etree.HTML(response_text)
        jp2_url_list = html.xpath('/html/body/table/tbody/tr[position()>1]/td[1]/a[contains(@href,"jp2")]')
        pdf_url_list = html.xpath('/html/body/table/tbody/tr[position()>1]/td[1]/a[contains(@href,"pdf")]')

        for (jp2,pdf) in zip(jp2_url_list, pdf_url_list):
            jp2_url = detail_url + jp2.xpath('./text()')[0]
            pdf_url = detail_url + pdf.xpath('./text()')[0]
            jp2_filename = "_".join(jp2_url.split("/")[-2:])
            pdf_filename = "_".join(pdf_url.split("/")[-2:])
            print(jp2_filename, pdf_filename)
            self.count += 1
            print(f"detail_url:{self.count} th data")
            response_content = requests.get(jp2_url, headers = self.headers).content
            with open(f'./jp2/{jp2_filename}', 'ab')as jp2_f:
                jp2_f.write(response_content)
            response_content = requests.get(pdf_url, headers = self.headers).content
            with open(f'./pdf/{pdf_filename}','ab')as pdf_f:
                pdf_f.write(response_content)

    # run program
    async def run(self):
        print('-------------begin crawling------------')
        pages_text = self.get_page()
        page3_urls = self.page_data_parse(pages_text)
        async with aiohttp.ClientSession() as session:
            tasks = [asyncio.create_task(self.save_data(detail_url,session)) for detail_url in page3_urls[self.count:]]
            await asyncio.wait(tasks)

if __name__ == '__main__':
    data = DateUrl()
    asyncio.run(data.run())

