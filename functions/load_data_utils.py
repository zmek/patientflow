def get_specialty_group(l):
    if (
        l.count("Medicine") > 0
        or l.count("Geriatric")
        or l.count("Cardiology")
        or l.count("Pharmacology ") > 0
        or l.count("Pharmacy ") > 0
        or l.count("Endocrinology") > 0
        or l.count("Rheumatology") > 0
        or l.count("Gastroenterology") > 0
        or l.count("Infectious Diseases") > 0
        or l.count("General Practice") > 0
        or l.count("Dermatology") > 0
    ):
        return "medical"
    elif (
        l.count("Surgery") > 0
        or l.count("Anaesthetics") > 0
        or l.count("Maternity") > 0
        or l.count("Obstetrics") > 0
        or l.count("Orthopaedics") > 0
        or l.count("Otolaryngology") > 0
        or l.count("Urology") > 0
        or l.count("Dental") > 0
    ):
        return "surgical"
    elif l.count("Oncology") > 0 or l.count("Haematology") > 0:
        return "haem_onc"
    return "medical"
