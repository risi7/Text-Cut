import numpy
import math
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords
import gensim
from gensim.summarization import summarize
from transformers import pipeline
import torch

#original_text='Junk foods taste good that’s why it is mostly liked by everyone of any age group especially kids and school going children. They generally ask for the junk food daily because they have been trend so by their parents from the childhood. They never have been discussed by their parents about the harmful effects of junk foods over health. According to the research by scientists, it has been found that junk foods have negative effects on the health in many ways. They are generally fried food found in the market in the packets. They become high in calories, high in cholesterol, low in healthy nutrients, high in sodium mineral, high in sugar, starch, unhealthy fat, lack of protein and lack of dietary fibers. Processed and junk foods are the means of rapid and unhealthy weight gain and negatively impact the whole body throughout the life. It makes able a person to gain excessive weight which is called as obesity. Junk foods tastes good and looks good however do not fulfil the healthy calorie requirement of the body. Some of the foods like french fries, fried foods, pizza, burgers, candy, soft drinks, baked goods, ice cream, cookies, etc are the example of high-sugar and high-fat containing foods. It is found according to the Centres for Disease Control and Prevention that Kids and children eating junk food are more prone to the type-2 diabetes. In type-2 diabetes our body become unable to regulate blood sugar level. Risk of getting this disease is increasing as one become more obese or overweight. It increases the risk of kidney failure. Eating junk food daily lead us to the nutritional deficiencies in the body because it is lack of essential nutrients, vitamins, iron, minerals and dietary fibers. It increases risk of cardiovascular diseases because it is rich in saturated fat, sodium and bad cholesterol. High sodium and bad cholesterol diet increases blood pressure and overloads the heart functioning. One who like junk food develop more risk to put on extra weight and become fatter and unhealthier. Junk foods contain high level carbohydrate which spike blood sugar level and make person more lethargic, sleepy and less active and alert. Reflexes and senses of the people eating this food become dull day by day thus they live more sedentary life. Junk foods are the source of constipation and other disease like diabetes, heart ailments, clogged arteries, heart attack, strokes, etc because of being poor in nutrition. Junk food is the easiest way to gain unhealthy weight. The amount of fats and sugar in the food makes you gain weight rapidly. However, this is not a healthy weight. It is more of fats and cholesterol which will have a harmful impact on your health. Junk food is also one of the main reasons for the increase in obesity nowadays.This food only looks and tastes good, other than that, it has no positive points. The amount of calorie your body requires to stay fit is not fulfilled by this food. For instance, foods like French fries, burgers, candy, and cookies, all have high amounts of sugar and fats. Therefore, this can result in long-term illnesses like diabetes and high blood pressure. This may also result in kidney failure. Above all, you can get various nutritional deficiencies when you don’t consume the essential nutrients, vitamins, minerals and more. You become prone to cardiovascular diseases due to the consumption of bad cholesterol and fat plus sodium. In other words, all this interferes with the functioning of your heart. Furthermore, junk food contains a higher level of carbohydrates. It will instantly spike your blood sugar levels. This will result in lethargy, inactiveness, and sleepiness. A person reflex becomes dull overtime and they lead an inactive life. To make things worse, junk food also clogs your arteries and increases the risk of a heart attack. Therefore, it must be avoided at the first instance to save your life from becoming ruined.The main problem with junk food is that people don’t realize its ill effects now. When the time comes, it is too late. Most importantly, the issue is that it does not impact you instantly. It works on your overtime; you will face the consequences sooner or later. Thus, it is better to stop now.You can avoid junk food by encouraging your children from an early age to eat green vegetables. Their taste buds must be developed as such that they find healthy food tasty. Moreover, try to mix things up. Do not serve the same green vegetable daily in the same style. Incorporate different types of healthy food in their diet following different recipes. This will help them to try foods at home rather than being attracted to junk food.In short, do not deprive them completely of it as that will not help. Children will find one way or the other to have it. Make sure you give them junk food in limited quantities and at healthy periods of time. '


def deep(original_text):
    summarization = pipeline('summarization')
    summary = summarization(original_text)[0]['summary_text']
    return summary


def tr(original_text):
    l=original_text.split('.')
    if(len(l)>10):
        summary = summarize(original_text)
    elif(len(l)>3 and len(l)<=10):
        summary = summarize(original_text,2)
    else:
        summary = original_text
    return summary

def ti(original_text):
    sentences = sent_tokenize(original_text) # NLTK function
    total_documents = len(sentences)
    freq_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()
    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        freq_matrix[sent] = freq_table
        tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(word_per_doc_table[word]))

        idf_matrix[sent] = idf_table

    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table


    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    sumValues = 0

    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))
    l=[]
    if(total_documents<5):
        for i in sentenceValue:
            if(sentenceValue[i]>=average):
                l.append(i)
        s = ' '.join(l)
    else:
        for i in sentenceValue:
            if(sentenceValue[i]>average):
                l.append(i)
        s = ' '.join(l)
    return s


def lr(original_text):
    threshold = 0.1
    epsilon = 0.1
    stopWords = set(stopwords.words("english"))
    sentences = sent_tokenize(original_text) # NLTK function
    ts = len(sentences)

    freq_matrix = {}

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            if word in stopWords:
                continue
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        freq_matrix[sent] = freq_table
    tf_matrix = {}
    c = 0
    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[c] = tf_table
        c = c+1

    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1
    idf_matrix = {}
    idf_table = {}

    for sent, f_table in freq_matrix.items():

        for word in f_table.keys():
            if word not in idf_table:
                idf_table[word] = math.log10(ts / float(word_per_doc_table[word]))
            else:
                pass

        idf_matrix[sent] = idf_table

    matrix = numpy.zeros((ts, ts))

    s = {}


    for i in range(ts):
        s[i] = sentences[i]
    stop_words = set(stopwords.words("english"))
    for i in range(ts):
        for j in range(ts):
            u1 = word_tokenize(s[i]) 
            u1 = [word for word in u1 if word.isalpha()]
            u1 = [word for word in u1 if word not in stop_words]
            u1 = [element.lower() for element in u1]
            u2 = word_tokenize(s[j]) 
            u2 = [word for word in u2 if word.isalpha()]
            u2 = [word for word in u2 if word not in stop_words]
            u2 = [element.lower() for element in u2]
            common = list(set(u1) & set(u2))
            d1=0
            d2=0
            n=0.0
            for t in u1:
                if t in tf_matrix[i]:
                    tf1=tf_matrix[i][t]
                else:
                    tf1=0
                if t in idf_table:
                    idf1=idf_table[t]
                else:
                    idf1=0
                d1+=(tf1*idf1)**2
            for t in u2:
                if t in tf_matrix[j]:
                    tf2=tf_matrix[j][t]
                else:
                    tf2=0
                if t in idf_table:
                    idf2=idf_table[t]
                else:
                    idf2=0
                d2+=(tf2*idf2)**2
            for t in common:
                if t in tf_matrix[i]:
                    tfc1=tf_matrix[i][t]
                else:
                    tfc1=0
                if t in tf_matrix[j]:
                    tfc2=tf_matrix[j][t]
                else:
                    tfc2=0
                if t in idf_table:
                    idf=idf_table[t]
                else:
                    idf=0
                n+=tfc1*tfc2*idf**2
            if d1 > 0 and d2 > 0:
                matrix[i][j] =  n / (math.sqrt(d1) * math.sqrt(d2))
            else:
                matrix[i][j] = 0.0
    degrees = numpy.zeros((ts, ))
    for i in range(ts):
        for j in range(ts):
            if matrix[i, j] > threshold:
                matrix[i, j] = 1.0
                degrees[i] += 1
            else:
                matrix[i, j] = 0

    for i in range(ts):
        for j in range(ts):
            if degrees[i] == 0:
                degrees[i] = 1

            matrix[i][j] = matrix[i][j] / degrees[i]

    transposed_matrix = matrix.T
    p_vector = numpy.array([1.0 / ts] * ts)
    lambda_val = 1.0

    while lambda_val > epsilon:
        next_p = numpy.dot(transposed_matrix, p_vector)
        lambda_val = numpy.linalg.norm(numpy.subtract(next_p, p_vector))
        p_vector = next_p
    
    avg = numpy.sum(p_vector) / len(p_vector)

    l=[]
    if(ts<7):
        for i in range(ts):
            if(p_vector[i]>=avg):
                l.append(s[i])
        sm = ' '.join(l)
    else:
        for i in range(ts):
            if(p_vector[i]>avg):
                l.append(s[i])
        sm = ' '.join(l)
    return sm



