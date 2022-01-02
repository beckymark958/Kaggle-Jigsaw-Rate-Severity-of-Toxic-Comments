
## -- Data Cleaning -- ##
def clean(col):
    
    # data[col] = data[col].str.replace('https?://\S+|www\.\S+', ' social medium ')      
        
    col = col.lower()
    col = col.replace("4", "a")
    col = col.replace("2", "l")
    col = col.replace("5", "s")
    col = col.replace("1", "i")
    col = col.replace("!", "i")
    col = col.replace("|", "i")
    col = col.replace("0", "o")
    col = col.replace("l3", "b")
    col = col.replace("7", "t")
    col = col.replace("7", "+")
    col = col.replace("8", "ate")
    col = col.replace("3", "e")
    col = col.replace("9", "g")
    col = col.replace("6", "g")
    col = col.replace("@", "a")
    col = col.replace("$", "s")
    col = col.replace("#ofc", " of fuckin course ")
    col = col.replace("fggt", " faggot ")
    col = col.replace("your", " your ")
    col = col.replace("self", " self ")
    col = col.replace("cuntbag", " cunt bag ")
    col = col.replace("fartchina", " fart china ")
    col = col.replace("youi", " you i ")
    col = col.replace("cunti", " cunt i ")
    col = col.replace("sucki", " suck i ")
    col = col.replace("pagedelete", " page delete ")
    col = col.replace("cuntsi", " cuntsi ")
    col = col.replace("i'm", " i am ")
    col = col.replace("offuck", " of fuck ")
    col = col.replace("centraliststupid", " central ist stupid ")
    col = col.replace("hitleri", " hitler i ")
    col = col.replace("i've", " i have ")
    col = col.replace("i'll", " sick ")
    col = col.replace("fuck", " fuck ")
    col = col.replace("f u c k", " fuck ")
    col = col.replace("shit", " shit ")
    col = col.replace("bunksteve", " bunk steve ")
    col = col.replace('wikipedia', ' social medium ')
    col = col.replace("faggot", " faggot ")
    col = col.replace("delanoy", " delanoy ")
    col = col.replace("jewish", " jewish ")
    col = col.replace("sexsex", " sex ")
    col = col.replace("allii", " all ii ")
    col = col.replace("i'd", " i had ")
    col = col.replace("'s", " is ")
    col = col.replace("youbollocks", " you bollocks ")
    col = col.replace("dick", " dick ")
    col = col.replace("cuntsi", " cuntsi ")
    col = col.replace("mothjer", " mother ")
    col = col.replace("cuntfranks", " cunt ")
    col = col.replace("ullmann", " jewish ")
    col = col.replace("mr.", " mister ")
    col = col.replace("aidsaids", " aids ")
    col = col.replace("njgw", " nigger ")
    col = col.replace("wiki", " social medium ")
    col = col.replace("administrator", " admin ")
    col = col.replace("gamaliel", " jewish ")
    col = col.replace("rvv", " vanadalism ")
    col = col.replace("admins", " admin ")
    col = col.replace("pensnsnniensnsn", " penis ")
    col = col.replace("pneis", " penis ")
    col = col.replace("pennnis", " penis ")
    col = col.replace("pov.", " point of view ")
    col = col.replace("vandalising", " vandalism ")
    col = col.replace("cock", " dick ")
    col = col.replace("asshole", " asshole ")
    col = col.replace("youi", " you ")
    col = col.replace("afd", " all fucking day ")
    col = col.replace("sockpuppets", " sockpuppetry ")
    col = col.replace("iiprick", " iprick ")
    col = col.replace("penisi", " penis ")
    col = col.replace("warrior", " warrior ")
    col = col.replace("loil", " laughing out insanely loud ")
    col = col.replace("vandalise", " vanadalism ")
    col = col.replace("helli", " helli ")
    col = col.replace("lunchablesi", " lunchablesi ")
    col = col.replace("special", " special ")
    col = col.replace("ilol", " i lol ")
    col = col.replace(r'\b[uU]\b', 'you')
    col = col.replace(r"what's", "what is ")
    col = col.replace(r"\'s", " is ")
    col = col.replace(r"\'ve", " have ")
    col = col.replace(r"can't", "cannot ")
    col = col.replace(r"n't", " not ")
    col = col.replace(r"i'm", "i am ")
    col = col.replace(r"\'re", " are ")
    col = col.replace(r"\'d", " would ")
    col = col.replace(r"\'ll", " will ")
    col = col.replace(r"\'scuse", " excuse ")
    col = col.replace('\s+', ' ')  # will remove more than one whitespace character
    col = col.replace(r'(.)\1+', r'\1\1') # 2 or more characters are replaced by 2 characters
    col = col.replace("[:|♣|'|§|♠|*|/|?|=|%|&|-|#|•|~|^|>|<|►|_]", '')
    
    col = col.replace(r"what's", "what is ")
    col = col.replace(r"\'ve", " have ")
    col = col.replace(r"can't", "cannot ")
    col = col.replace(r"n't", " not ")
    col = col.replace(r"i'm", "i am ")
    col = col.replace(r"\'re", " are ")
    col = col.replace(r"\'d", " would ")
    col = col.replace(r"\'ll", " will ")
    col = col.replace(r"\'scuse", " excuse ")
    col = col.replace(r"\'s", " ")

    # Clean some punctutations
    col = col.replace('\n', ' \n ')
    col = col.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3')
    # Replace repeating characters more than 3 times to length of 3
    col = col.replace(r'([*!?\'])\1\1{2,}',r'\1\1\1')    
    # Add space around repeating characters
    col = col.replace(r'([*!?\']+)',r' \1 ')
    # patterns with repeating characters 
    col = col.replace(r'([a-zA-Z])\1{2,}\b',r'\1\1')
    col = col.replace(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1')
    col = col.replace(r'[ ]{2,}',' ').strip()
    col = col.replace(r'[ ]{2,}',' ').strip()
    return col
