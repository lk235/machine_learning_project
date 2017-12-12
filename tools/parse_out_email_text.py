#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    stemmer = SnowballStemmer('english')
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### project part 2: comment out the line below
        # words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)

        # print text_string
        # text_string.replace('\n',' ')
        # for i in ["sara","shackleton","chris","germani"]:
        #     text_string = text_string.replace(i,'')

        sp = text_string.split()
        # print sp
        # print sp[0]

        for i in sp:
            # i = i.replace('\n\n',' ')
            # print i
            # if i != '':
            si = stemmer.stem(i)
            words = words + si + ' '
        # print text_string


    return words

    

def main():

    # ff = open("../text_learning/test_email.txt", "r")
    ff = open("../enron_mail_20150507\maildir/jones-t/all_documents/4046.", "r")
    text = parseOutText(ff)
    print text



if __name__ == '__main__':
    main()

