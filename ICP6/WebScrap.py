import bs4 as bs
import urllib.request

source = urllib.request.urlopen('https://en.wikipedia.org/wiki/Wasim_Akram').read()
soup = bs.BeautifulSoup(source,'lxml')
for paragraph in soup.find_all('p'):
    print(str(paragraph.text))
    outF = open("myOutFile.txt", "a")
    for line in str(paragraph.text):
        outF.write(line)
    outF.close()
