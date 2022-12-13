
f= open ('error.txt')
text=f.read()
import nltk

def VerbFormError(tags):
    if tags[2].startswith('V') and tags[2] != 'VB':
        if tags[1] == 'MD' or tags [1] == 'TO':
            return True
        elif tags[0] == 'MD' or tags [0] == 'TO':
            return True
        else:
            return False
    else:
        return False

def SubjectVerbAgreementError(tags):
    if tags[2].startswith('V') and tags[2] == 'VBP':
        if tags[1] == 'NN' or tags [1] == 'NNP':
            return True
        elif tags[0] == 'NN' or tags [0] == 'NNP':
            return True
        else:
            return False
    elif tags[2].startswith('V') and tags[2] == 'VBZ':
        if tags[1] == 'NNS' or tags [1] == 'NNPS':
            return True
        elif tags[0] == 'NNS' or tags [0] == 'NNPS':
            return True
        else:
            return False
    elif tags[2].startswith('V') and tags[2] == 'VB':
        if tags[1] == 'PRP' and tags[2] != 'MD':
            return True
        else: 
            return False
    else:
        return False

    

sents= nltk.sent_tokenize(text)

import string
wordlist=nltk.corpus.words.words()
wnl=nltk.WordNetLemmatizer()

for sent in sents:
    words=nltk.word_tokenize(sent)
    tagged_words=nltk.pos_tag(words)
    curTags=['<tag>', '<tag>', '<tag>']
    curWords= ['<word>', '<word>', '<word>']
    print (sent)
    POS=''


    if words[0] [0]. islower():
        print ('** Capitalization error:', words[0] )
        
            
    for tagged_word in tagged_words:
        curWord = tagged_word [0]
        curPOS = tagged_word[1]            
        curTags.pop(0)
        curTags.append(curPOS)
        curWords.pop(0)
        curWords.append(curWord)
        
        if curPOS.startswith('V') :
            lemma =wnl.lemmatize(curWord, 'v')
        
        elif curPOS.startswith('N') :
            lemma =wnl.lemmatize(curWord, 'n')
        
            if (not(lemma in wordlist)) and (not (lemma in string.punctuation )) and curWord.islower():           
                print('** Spelling error:', curWord)
  
        if VerbFormError(curTags):
            print ('**Verb form error:', curWord)
            
        
        if SubjectVerbAgreementError(curTags):
            print ('**Subject-verb agreement error:', curWord)
      
        
        POS=POS+tagged_word[1]
    if not 'V' in POS:
        print ('** Fragment error')
       
    

f.close()        
    
    
    
   
  
    
    





# %%
