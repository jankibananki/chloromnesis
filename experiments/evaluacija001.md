Nakon prvog uspesnog treninga, odradjena je provera koliko dobro model razlikuje pet kategorija bolesti kod paradajza.
    Ucitan je prethodno sacuvani model chloromnesis_v1.pth i pokrenuta je evalucija nad 783 fotografija iz validacionog skupa bez dodatnog treniranja. Za analizu je koriscen classification_report i confusion_matrix iz biblioteke scikit_learn. 

Rezultati: 
    Ukupna tačnost: 91.19% (714/783)
    Macro F1-score: 0.909

Model generalno daje obecavajuce rezultate - najvise gresi kod Early Blight-a: od 160 fotografija model je pravilno prepoznao 114 a pogresio na 46. Najcesce ga mesa sa: 
    Late Blight — 15 fotografija
    Leaf Mold — 13 fotografija
    Healthy — 9 fotografija
    Septoria Leaf Spot — 9 fotografija