
import urllib.request
import json

link = "https://klimalog.die-gdi.de/ndc/open-data/dataset.json"
def get_document(country_code: str):
    """
    read the country NDC data from 
    https://klimalog.die-gdi.de/ndc/open-data/dataset.json 
    using the country code.
    
    Params
    -------
    country_code:"""
    with urllib.request.urlopen(link) as urlfile:
        data =  json.loads(urlfile.read())
    categoriesData = {}
    categoriesData['categories']= data['categories']
    categoriesData['subcategories']= data['subcategories']
    keys_sub = categoriesData['subcategories'].keys()
    documentType= 'NDCs'
    if documentType in data.keys():
        if country_code in data[documentType].keys():
            get_dict = {}
            for key, value in data[documentType][country_code].items():
                if key not in ['country_name','region_id', 'region_name']:
                    get_dict[key] = value['classification']
                else:
                    get_dict[key] = value
        else:
            return None
    else:
        return None

    country = {}
    for key in categoriesData['categories']:
        country[key]= {}
    for key,value in categoriesData['subcategories'].items():
        country[value['category']][key] = get_dict[key]
    
    return country
        
            
def countrySpecificCCA(cca_sent:dict, threshold:int, countryCode:str):
    """
    based on the countrycode, reads the country data from
    https://klimalog.die-gdi.de/ndc/open-data/dataset.json
    using get_documents from utils.ndc_explorer.py
    then based on thereshold value filters the Climate Change Adaptation
    targets assigned by NDC explorer team to that country. Using the sentences
    create by Data services team of GIZ for each target level, tries to find the
    relevant passages from the document by doing the semantic search.

    Params
    -------
    cca_sent: dictionary with key as 'target labels' and manufactured sentences 
    reflecting the target level. Please see the docStore/ndcs/cca.txt

    threshold: NDC target have many categoriees ranging from [0-5], with 0 
    refelcting most relaxed attitude and 5 being most aggrisive towards Climate 
    change. We select the threshold value beyond which we need to focus on.

    countryCode: standard country code to allow us to fetch the country specific
    data.

    """
    temp = {}
    doc = get_document(countryCode)
    for key,value in cca_sent.items():
        id_ = doc['climate change adaptation'][key]['id']
        if id_ >threshold:
            temp[key] = value['id'][id_]
    return temp

                
def countrySpecificCCM(ccm_sent, threshold, countryCode):
    """
    see the documentation of countrySpecificCCA. This is same instead of 
    this gets the data pertaining to Adaptation
    
    """

    temp = {}
    doc = get_document(countryCode)
    for key,value in ccm_sent.items():
        id_ = doc['climate change mitigation'][key]['id']
        if id_ >threshold:
            temp[key] = value['id'][id_]
    
    return temp
