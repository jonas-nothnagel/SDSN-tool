import streamlit as st

def app():
    
    
    with open('style.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center;  \
                      color: black;'> Climate Policy Action Tracker Manual</h2>", 
                      unsafe_allow_html=True)

    
    st.markdown("<div style='text-align: center; \
                    color: grey;'>The Policy Action Tracker is an open-source\
                         digital tool which aims to assist policy analysts and \
                          other users in extracting and filtering relevant \
                            information from public documents.</div>",
                        unsafe_allow_html=True)
    footer = """
           <div class="footer-custom">
               Guidance & Feedback - <a href="https://www.linkedin.com/in/maren-bernlöhr-149891222" target="_blank">Maren Bernlöhr</a> |
               <a href="https://www.linkedin.com/in/manuelkuhm" target="_blank">Manuel Kuhm</a> |
               Developer - <a href="https://www.linkedin.com/in/erik-lehmann-giz/" target="_blank">Erik Lehmann</a>  |   
               <a href="https://www.linkedin.com/in/jonas-nothnagel-bb42b114b/" target="_blank">Jonas Nothnagel</a>   |
               <a href="https://www.linkedin.com/in/prashantpsingh/" target="_blank">Prashant Singh</a> |
               
           </div>
       """
    st.markdown(footer, unsafe_allow_html=True)

    c1, c2, c3 =  st.columns([8,1,12])
    with c1:
        st.image("docStore/img/ndc.png")
    with c3:
        st.markdown('<div style="text-align: justify;">The manual extraction \
        of relevant information from text documents is a \
    time-consuming task for any policy analysts. As the amount and length of \
    public policy documents in relation to sustainable development (such as \
    National Development Plans and Nationally Determined Contributions) \
    continuously increases, a major challenge for policy action tracking – the \
    evaluation of stated goals and targets and their actual implementation on \
    the ground – arises. Luckily, Artificial Intelligence (AI) and Natural \
    Language Processing (NLP) methods can help in shortening and easing this \
    task for policy analysts.</div><br>',
    unsafe_allow_html=True)

    intro = """
    <div style="text-align: justify;">

    For this purpose, the United Nations Sustainable Development Solutions \
    Network (SDSN) and the Deutsche Gesellschaft für Internationale \
    Zusammenarbeit (GIZ) GmbH are collaborating since 2021 in the development \
    of an AI-powered open-source web application that helps find and extract \
    relevant information from public policy documents faster to facilitate \
    evidence-based decision-making processes in sustainable development and beyond.  

    The collaboration aims to determine the potential of NLP methods for \
        tracking policy implementation and coherence in the context of the \
        Sustainable Development Goals (SDGs) and the Paris Climate Agreement. \
        Nationally determined contributions (NDCs) will serve as a starting \
        point for the analysis and evaluation in a specific national context. \
        Under the Paris Climate Agreement, NDCs embody the efforts of each \
        country to reduce national emissions and thus contribute to the \
        achievement of the long-term goals of the Agreement – to increase the \
        ability to adapt to adverse impacts of climate change and foster \
        climate resilience and low greenhouse gas emissions development, in a \
        manner that does not threaten food production. The Paris Climate \
        Agreement (Article 4, Paragraph 2)1 requires each Party to prepare, \
        communicate and maintain successive NDCs. Thus, they serve as a \
        comparable, accessible, and widely acknowledged starting point for \
        analysis. However, the agreed and communicated goals and measures must \
        also be reflected in national strategies, statements, and other \
        government publications to be implemented timely, as well as effectively.\
        At best, the activities and measures should have an allocated budget. \
        Given the complexity, the manual evaluation of policy documents and \
        other publications has been very time-consuming and has presented a \
        significant challenge for policy analysts and makers alike. In \
        consequence, the open-source web application aims to support the process\
         through suitable AI-powered and NLP methods. In the following, the \
        application’s functionalities are explained in more detail.
    </div>
    <br>
    """
    st.markdown(intro, unsafe_allow_html=True)
    st.image("docStore/img/pic1.png")