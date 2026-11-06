Inicijalno treniranje modela - uspesno!

Koriscen je vec istreniran model MobileNetV3 Small koji je prilagodjen projektu pomocu transfer learning-a.
Cilj: da model na osnovu fotografije lista paradajza prepozna jednu od pet kategorija bolesti.

Sto se podataka tice, koriscen je PlantVillage dataset. Za prvi eksperiment izdvojeno je 3903 fotografija za koje je pronadjen identifikator lista.

    -Training: 3120 fotografija
    -Validation: 783 fotografije

Podaci su podeljeni pomocu StratifiedGroupKFold da se fotografije istog lista ne bi nasle i u trening skupu i u validacionom skupu.
    Provera je pokazala 0 zajednickih listova.
Fotografije bez poznatog leaf_id su preskocene.

    Model: MobileNetV3 Small
    Pristup: Transfer Learning
    Loss: CrossEntropyLoss
    Optimizator: AdamW
    Learning rate 0.001
    Batch size: 16
    Broj epoha: 1
    Uredjaj: cpu

Rezultati: 
    Loss: 
            -Training: 0.4198
            -Validation: 0.2692
    Accuracy:
            -Training: 84.94%
            -Validation: 91.19%

Tokom treninga accuracy je postepeno rastao, dok se loss smanjivao, sto nam pokazuje da je model poceo da uci. 

Prvi trening je uspeo ali jos uvek ne znamo koje bolesti model najcesce mesa.