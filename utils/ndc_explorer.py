
import urllib.request
import json

link = "https://klimalog.die-gdi.de/ndc/open-data/dataset.json"
def get_document(countryCode: str):
                with urllib.request.urlopen(link) as urlfile:
                    data =  json.loads(urlfile.read())
                categoriesData = {}
                categoriesData['categories']= data['categories']
                categoriesData['subcategories']= data['subcategories']
                keys_sub = categoriesData['subcategories'].keys()
                documentType= 'NDCs'
                if documentType in data.keys():
                    if countryCode in data[documentType].keys():
                        get_dict = {}
                        for key, value in data[documentType][countryCode].items():
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
        
        #   country_ndc = get_document('NDCs', countryList[option])
            
def countrySpecificCCA(cca_sent, threshold, countryCode):
    temp = {}
    doc = get_document(countryCode)
    for key,value in cca_sent.items():
        id_ = doc['climate change adaptation'][key]['id']
        if id_ >threshold:
            temp[key] = value['id'][id_]
    return temp

                
def countrySpecificCCM(ccm_sent, threshold, countryCode):
    temp = {}
    doc = get_document(countryCode)
    for key,value in ccm_sent.items():
        id_ = doc['climate change mitigation'][key]['id']
        if id_ >threshold:
            temp[key] = value['id'][id_]
    
    return temp
