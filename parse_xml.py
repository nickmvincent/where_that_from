import xml.etree.ElementTree


def parse_xml():
    filename = 'sample_paper.xml'
    e = xml.etree.ElementTree.parse(filename).getroot()
    print(e)
    out = ""

    for page in e.findall('page'):
        for region in page.findall('region'):
            out += region.text + ' '
            print(region.text)
    with open(filename.replace('.xml', '.txt'), 'w') as f:
        f.write(out)

if __name__ == '__main__':
    parse_xml()