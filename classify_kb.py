#run SVM on songs based on features derived from notes
import pickle
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn import svm,tree
#import pydotplus
import random

def generate_features(songs,matches,chords,first=False,last=False,relative=False,major=False,minor=False):
    #generate features for each song in matches
    #relative flag transforms absolute chromatic scale [C..Bm] into relative scale [Root..Seventh Minor]
    #applies to the prediction task over ALL 24 keys and to major/minor prediction
    feature_list=[]
    labels=[]
    matched_songs=[]
    for song in songs:
        if song in matches:
            features=[0]*24 #feature array per song
            total=len(songs[song]) #total number of chords for normalization
            root=matches[song] #this is the real key (our label)
            if major and 'm' in root:
                continue #we only want major keys
            if minor and not 'm' in root:
                continue #we only want minor keys
            transform=(chords[root]*relative) % 12 #this is our transformation based on the root in relative mode; multiply by boolean relative to set transform equal to actual number or 0 and take modulus 12 to get in range [0,11]
            #print("root is",root,"transform is",transform)
            for chord in songs[song]:
                features[transform_chord(chord,transform,chords)]+=1
            features=[feature/total for feature in features]
            #consider alternate features
            #NOTE: SVM needs floats so use 24-base index rather than note name
            if first:
                features.append(transform_chord(songs[song][0],transform,chords))
            if last:
                features.append(transform_chord(songs[song][-1],transform,chords))
            #append to parallel arrays
            feature_list.append(features)
            #labels.append(chords[root])
            #predict major/minor task
            if 'm' in root: labels.append(0)
            else: labels.append(1)
            matched_songs.append(song)
    print("last root",root,"last transform",transform)
    print("Length of feature vector: ",len(features))
    return (feature_list,labels,matched_songs)

def transform_chord(chord,transform,chords):
    #transform chord at index based on relative key
    if 'm' in chord: #minor chord
        return (chords[chord]-transform) % 12 + 12
    else: #major chord
        return (chords[chord]-transform) % 12

def averageDistribution(songs,matches,chords,major=False,minor=False):
    #find average relative distributions of chord data
    #set major/minor to gather only major/minor keys (mutually exclusive)
    average_dist=[0]*24
    for song in songs:
        if song in matches:
            features=[0]*24 #feature array per song
            total=len(songs[song]) #total number of chords for normalization
            transform=chords[matches[song]] % 12
            if major:
                if chords[matches[song]]>=12: #minor key
                    continue
            if minor:
                if chords[matches[song]]<12: #major key
                    continue
            for chord in songs[song]:
                features[transform_chord(chord,transform,chords)]+=1
            features=[feature/total for feature in features]
            for i in range(len(features)):
                average_dist[i]+=features[i]
    average_dist=[average/len(matches) for average in average_dist]
    return average_dist


def find_diffs(feature_list,labels,major_average,minor_average):
    #turn normalized features into differences from average distributions
    for i in range(len(feature_list)):
        if labels[i]: #major key
            feature_list[i]=diff(feature_list[i],major_average)
        else: #minor key
            feature_list[i]=diff(feature_list[i],minor_average)
    return feature_list

def diff(features,average_dist):
    #diff feature_vector from average distribution
    return [features[i]-average_dist[i] for i in range(len(features))]


def generate_rel_features(song,matches,chords,first=False,last=False,major=False,minor=False,relative=True):

    feature_list=[]
    labels=[]
    matched_songs=[] #will be redundant, one entry per root
    roots=[] #additionally return list of roots
    for song in songs:
        if song in matches:
            total=len(songs[song]) #total number of chords for normalization
            #try out all chords in the song as our root; one should be correct
            label=matches[song] #this is the real key (our label)
            if major and 'm' in label:
                continue #we only want major keys
            if minor and not 'm' in label:
                continue #we only want minor keys
            for root in set(songs[song]):                
                features=[0]*24
                transform=(chords[root]*relative) % 12 
                #print("root is",root,"transform is",transform)
                for chord in songs[song]:
                    features[transform_chord(chord,transform,chords)]+=1
                features=[feature/total for feature in features]
                #consider alternate features
                #NOTE: SVM needs floats so use 24-base index rather than note name
                if first:
                    features.append(transform_chord(songs[song][0],transform,chords))
                if last:
                    features.append(transform_chord(songs[song][-1],transform,chords))
                #append to parallel arrays
                feature_list.append(features)
                #our label is either this distribution is valid or it isn't
                #check index in chords to account for different chord names
                labels.append(chords[root]==chords[label])
                matched_songs.append(song)
                roots.append(root)
                #print(root,features,labels[-1],label)

            #print("our root is",label,labels)
    print("last root",root,"last transform",transform)
    print("Length of feature vector: ",len(features))
    return (feature_list,labels,matched_songs,roots)


def predict_most_frequent(songs,matches):
    #predict key based on most frequent chord
    correct=0
    for song in songs:
        if song in matches:
            label=matches[song]
            if label==max(songs[song],key=songs[song].count):
                correct+=1
    return correct/len(matches)

def relative_chords(lst,root,chords):
    #return normalized distribution of chord list based on root
    features=[0]*24
    transform=chords[root] % 12
    total=len(lst)
    for chord in lst:
        features[transform_chord(chord,transform,chords)]+=1
    return [feature/total for feature in features]
    
def test_chords(test_keys,chord_list,chords,clf_major,clf_minor,first=False,last=False):
    '''test all of the chords in a song to predict the top result; takes an iterable of the keys to test, a list containing all of the chords in the song, a list of all possible chords, a classifier for major chords and minor chords (may be the same), and first/last flags to include the first/last chord as a feature'''
    scores=[] #holds our prediction probabilities
    for chord in test_keys:
        #try all chords
        item=relative_chords(chord_list,chord,chords) #relativize
        transform=chords[chord] % 12
        if first:
            item.append(transform_chord(chord_list[0],transform,chords))
        if last:
            item.append(transform_chord(chord_list[-1],transform,chords))
        if 'm' in chord: #minor prediction
            scores.append((clf_minor.predict_proba([item]),chord)) #predict
        else:
            scores.append((clf_major.predict_proba([item]),chord))
        #print("predicting root: ",chord,score)
    #pull out confidence scores for positive classification
    scores=[(tup[0][0][1],tup[1]) for tup in scores]
    return sorted(scores,reverse=True)

def test_song(chord_list,chords,clf,chromatic,first=False,last=False,mode=False):
    '''test all of the chords in a song to predict the top result; takes a list containing all of the chords in the song, a list of all possible chords, a classifier for major chords and minor chords (may be the same), first/last flags to include the first/last chord as a feature, and a mode flag that can be set to true to do major/minor prediction'''
    scores=[] #holds our prediction probabilities

    item=relative_chords(chord_list,'C',chords) #relativize to C major

    if first:
        item.append(transform_chord(chord_list[0],0,chords))
    if last:
        item.append(transform_chord(chord_list[-1],0,chords))
        

    #if 'm' in chord: #minor prediction
    scores=clf.predict_proba([item])[0]
    #alternate for major/minor task
    if mode: #return scores before final processing
        return scores
    scores=[(scores[i],chromatic[i]) for i in range(len(scores))]
    #scores.append((clf_minor.predict_proba([item]),chord)) #predict
    #else:
    #    scores=clf_minor.predict_proba([item])[1]
    #    scores.append((clf_major.predict_proba([item]),chord))

    #pull out confidence scores for each of 24 labels
    #scores=[(tup[0][0][1],tup[1]) for tup in scores]
    return sorted(scores,reverse=True)

def generate_diff_features(feature_list,labels,major_average,minor_average):

    #turn normalized features into differences from average distributions
    for i in range(len(feature_list)):
        if 'm' in labels[i]: #minor key
            feature_list[i]=diff(feature_list[i],minor_average)
        else: #major key
            feature_list[i]=diff(feature_list[i],major_average)
    return feature_list


#comment out to run this program on its own
'''
#main section
print("Reading in data...\n")

#read in pickled dataset
with open("new_data.p","rb") as f:
    songs=pickle.load(f)

#find matches
matches={}
with open('matches.txt') as f:
    for line in f:
        line=line.strip().split(",")
        matches[int(line[3])]=line[2]


#specify equivalent note sets
chords={'C':0, 'B#':0, 'C#':1, 'Db':1, 'D':2, 'D#':3, 'Eb':3, 'Fb':4, 'E':4, 'E#':5, 'F':5, 'Gb':6, 'F#':6, 'G':7, 'G#':8, 'Ab':8, 'A':9, 'A#':10, 'Bb':10, 'B':11, 'Cb':11, 'Cm':12, 'B#m':12, 'C#m':13, 'Dbm':13, 'Dm':14, 'D#m':15, 'Ebm':15, 'Em':16, 'Fbm':16, 'Fm':17, 'E#m':17, 'Gbm':18, 'F#m':18, 'Gm':19, 'Abm':20, 'G#m':20, 'Am':21, 'A#m':22, 'Bbm':22, 'Cbm':23, 'Bm':23}

chromatic=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','Cm','C#m','Dm','D#m','Em','Fm','F#m','Gm','G#m','Am','A#m','Bm']
relative=['1','b2','2','b3','3','4','b5','5','b6','6','b7','7','1m','b2m','2m','b3m','3m','4m','b5m','5m','b6m','6m','b7m','7m']

#find baseline of predicting key by most frequently appearing chord
#print("accuracy for most frequent chords: ",predict_most_frequent(songs,matches))

#randomize matches
length=len(matches)
ids=list(matches.keys()) #a random list of all songs
random.shuffle(ids)
train=set([song for song in ids[:length//5*4]])
test=set([song for song in ids[length//5*4:]])
print("intersection between train and test sets: ",train.intersection(test))
#lst=[i for i in range(length)]
#random.shuffle(lst)
#train=[list(matches.keys())[i] for i in lst[:length//5*4]]
#test=[list(matches.keys())[i] for i in lst[length//5*4:]]
new_matches={}
for song in matches:
    if song in train:
        new_matches[song]=matches[song]

#generate list of features and labels
print("Generating feature list...\n")
#feature_list,labels,data=generate_features(songs,new_matches,chords,first=False,last=False,relative=False,major=False,minor=False)
feature_list,labels,data,roots=generate_rel_features(songs,new_matches,chords,first=False,last=False,major=False,minor=False,relative=True)
print(len(feature_list),"out of",len(matches),"matched")
#print(feature_list[-1],labels[-1],songs[data[-1]])

#print matched songs that have valid chords to a new file
#with open('matched_song_ids.txt','w') as f:
#    for songid in data:
#        f.write(str(songid)+'\n')

#run alternate classifier based on relative differences
#major_dist=averageDistribution(songs,matches,chords,major=True,minor=False)
#minor_dist=averageDistribution(songs,matches,chords,major=False,minor=True)
#feature_list=find_diffs(feature_list,labels,major_dist,minor_dist)
#print(feature_list[-1])
#with open("minor_average_dist.txt","w") as f:
#    for i in range(len(relative)):
#        f.write(relative[i]+","+str(average_dist[i])+"\n")

#change normalized features into differences from average distributions
#feature_list=generate_diff_features(feature_list,[matches[songid] for songid in data],major_dist,minor_dist)

#check labels
#negatives=[label for label in labels if not label]
#print("accuracy by predicting all negative: ",len(negatives)/len(labels))

#split randomly into training set/testing set
X_train,X_test,Y_train,Y_test=train_test_split(feature_list,labels,train_size=0.80)
#print(len(X_train),"in training set",len(X_test),"in testing set")
'''

#perform cross-validation with SVM
#clf=svm.SVC(kernel='linear',C=1)
#scores=cross_val_score(clf,X_train,Y_train,cv=5)
#print("Scores for cross validation: ",scores)

'''
#perform grid search to get best parameter values
print("Performing grid search...")
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
svr=svm.SVC()
clf=GridSearchCV(svr,param_grid)
clf.fit(X_train,Y_train)
print(clf.best_score_, clf.best_params_)
print(clf.scorer_)
print(clf.cv_results_['params'])
'''

'''
OUPUT FROM GRID SEARCH:
0.393006214637 {'kernel': 'rbf', 'C': 1000, 'gamma': 0.001}
<function _passthrough_scorer at 0x7f15b0c97c80>
({'kernel': 'linear', 'C': 1}, {'kernel': 'linear', 'C': 10}, {'kernel': 'linear', 'C': 10                0}, {'kernel': 'linear', 'C': 1000}, {'kernel': 'rbf', 'C': 1, 'gamma': 0.001}, {'kernel':                 'rbf', 'C': 1, 'gamma': 0.0001}, {'kernel': 'rbf', 'C': 10, 'gamma': 0.001}, {'kernel': '                rbf', 'C': 10, 'gamma': 0.0001}, {'kernel': 'rbf', 'C': 100, 'gamma': 0.001}, {'kernel': '                rbf', 'C': 100, 'gamma': 0.0001}, {'kernel': 'rbf', 'C': 1000, 'gamma': 0.001}, {'kernel':                 'rbf', 'C': 1000, 'gamma': 0.0001})

Best result an rbf kernel but this barely outperforms the default settings we had before
Without first/last chord:
kernel='linear',C=1: accuracy on test set: 0.376483679525
kernel='rbf',C=1000,gamma=0.001: accuracy on test set: 0.399480712166
Adding alternate features does not help...
With first chord and best parameters: 0.390578635015
With last chord and best parameters: 0.382047477745
With first and last chords and best parameters: 0.38649851632

RELATIVE keys perform much better:
Without first/last chord: 0.497403560831
With first chord: 0.501112759644
With last chord: 0.50296735905
With first and last chords: 0.520029673591

UPDATE:
Most frequent only: 0.2509460562439712

non-relative with correct labelings:
Without first/last chord: 0.381305637982
With first/last chord: 0.370919881306
relative with correct labelings:
Without first/last chord: 0.229599406528
With first/last chord: 0.247403560831

Major only:
non-relative:
Without first/last chord: 0.482382133995
With first/last chord: 0.489330024814
relative:
Without first/last chord: 0.270967741935
With first/last chord: 0.278908188586

Minor only:
non-relative:
Without first/last chord: 0.43906020558
With first/last chord: 0.462555066079
relative:
Without first/last chord: 0.325991189427
With first/last chord: 0.277533039648

non-relative forced prediction, manual testing split:
Without first/last chord: 0.40155728587319245
with first/last chord: 0.39154616240266965
major/minor seperate: 0.37226548016314426
with first/last chord: 0.3882091212458287

TASK - PREDICT MAJOR/MINOR:
non-relative:
Without first/last chord: 0.75296735905
With first/last chord: 0.763724035608
relative:
Without first/last chord: 0.831231454006
With first/last chord: 0.841246290801
relative differences to major/minor distributions:
Without first/last chord: 1.0

TASK - PREDICT VALID DISTRIBUTION:
non-relative:
Without first/last chord: 0.865623210536
major only: 0.862790697674
minor only: 0.901545095221
relative:
Without first/last chord: 0.871765514083
With first/last chord: 0.872834134799
major only: 0.852335093714
minor only: 0.8975029036

null model (predict all as negative):
without first/last chord: 0.8683089850090709
major only: 0.8543789767730589
minor only: 0.9086264778864399

non-relative:
forced prediction: 0.06377456433073786
over all 24 possibilities: 0.010752688172043012
with major/minor seperate: 0.08342602892102335
over all 24: 0.035224323322209865

relative:
forced prediction: 0.38561364479050797
over all 24 possibilities: 0.04375231738969225 (cannot differentiate major/minor)
With first/last chord: 0.41045606229143494
over all 24: 0.05005561735261402
major only: 0.37189469781238416
over all 24: 0.017426770485724878
minor only: 0.13385242862439747
over all 24: 0.10752688172043011
using both major/minor classifiers: 0.3971078976640712
over all 24: 0.39117538005190955
both major/minor with first/last chords: 0.3515016685205784
over all 24: 0.3348164627363737

with differences: 0.3129403040415276
over all 24: 0.10159436410826844
with major/minor seperate: 0.3967371153133111
over all 24: 0.3874675565443085

forced combos:
absolute key rank * relative rank: 0.38561364479050797
over all 24: 0.38079347423062665
absolute key rank * relative major/minor rank: 0.41156840934371525
over all 24: 0.40415276232851316

absolute key score * relative score: 0.3885799035965888
over all 24: 0.3793103448275862
absolute key score * relative major/minor score: 0.4175009269558769
over all 24: 0.41564701520207636

2-part: major/minor then relative: 0.38524286243974787
over all 24: 0.3659621802002225
END UPDATE

DECISION TREES:
absolute:
Without first/last chord: 0.229970326409
With first chord: 0.222551928783
With last chord: 0.231083086053
With first and last chords: 0.215504451039

relative:
Without first/last chord: 0.426557863501
With first chord: 0.439169139466
With last chord: 0.452151335312
With first and last chords: 0.43175074184
'''

'''
#run SVM classifier
#clf=svm.SVC(kernel='linear',C=1).fit(X_train,Y_train)
#clf=svm.SVC(kernel='rbf',C=1000,gamma=0.001).fit(X_train,Y_train)
#print(clf.score(X_test,Y_test))

#force a choice for each test data point; overwrite testing/training splits so taht we know ids
print("songs used for training set: ",len(train),";  songs used for testing set",len(test))
#print(feature_list[0],labels[0],data[0],roots[0],set(songs[data[0]]))

#train SVM; training set filtered above
#clf=svm.SVC(kernel='rbf',C=1000,gamma=0.001,probability=True).fit(feature_list,labels)
#save classifier
#with open('svm_mode.pkl','wb') as f:
#    s=pickle.dump(clf,f)
#exit()
#read in classifier
with open('svm_major.pkl','rb') as f:
    clf_major=pickle.load(f)

#read in classifier
with open('svm_minor.pkl','rb') as f:
    clf_minor=pickle.load(f)

#read in classifier
with open('svm_absolute.pkl','rb') as f:
    clf_abs=pickle.load(f)

#read in classifier
with open('svm_mode.pkl','rb') as f:
    clf_mode=pickle.load(f)

#predict one key per song
print("created model, testing keys...")
correct=0 #correct predictions
try_all=False #flag to try all 24 chromatic positions; if false, only try chords in songs
add_mode=False

for songid in test:

    label=matches[songid]
    #print("songid: ",songid,"true label is: ",label)
    #NOTE: a lot of keys do not actually appear in the song (mistake on Ultimate Guitar data?)
    if add_mode: #run mode classifier first to filter
        mode=list(test_song(songs[songid],chords,clf_mode,chromatic,first=False,last=False,mode=True))
        mode=mode.index(max(mode))

    if not try_all:
        test_keys=set(songs[songid])
        #scores=test_chords(set(songs[songid]),songs[songid],chords,clf_major,clf_minor,first=False,last=False)
    else: #try all 24 chromatic positions
        test_keys=chromatic
    #filter test_keys based on mode
    if add_mode and mode: #only major
        test_keys={key for key in test_keys if 'm' not in key}
    if add_mode and not mode: #only minor
        test_keys={key for key in test_keys if 'm' in key}

    scores=test_chords(test_keys,songs[songid],chords,clf_major,clf_minor,first=False,last=False)
    
    if 'm' in label: #minor key
        abs_scores=test_song(songs[songid],chords,clf_abs,chromatic,first=False,last=False)
    else: #major key
        abs_scores=test_song(songs[songid],chords,clf_abs,chromatic,first=False,last=False)
    #print(sorted(abs_scores))
    #print(max(abs_scores),label)
    
    #try:
    #    prediction=max(scores)[1]
    #except:
    #    print(songs[songid],mode,scores,test_keys)
    #    exit()
    
    #alternate test: combine absolute classifier with relative classifier
    rel_rank=[chords[score[1]] for score in scores]
    abs_rank=[chords[score[1]] for score in abs_scores]
    #print(songs[songid])
    #print(scores,abs_scores)
    #print(rel_rank,abs_rank)
    final_scores=[]
    
    #for chord in set(rel_rank).intersection(set(abs_rank)): #these may be different sizes
        #calculate rankproduct
        #final_scores.append(((rel_rank.index(chord)+1)*(abs_rank.index(chord)+1),chord))
 

    #alternate to use score rather than rankproduct
    for chord1 in scores:
        for chord2 in abs_scores:
            if chords[chord1[1]]==chords[chord2[1]]:
                final_scores.append((chord1[0]*chord2[0],chords[chord1[1]]))
    prediction=max(final_scores)[1]

    #print(final_scores)
    #prediction=min(final_scores)[1]
    #print(rel_rank,abs_rank)
    
    #if prediction is based on chromatic position
    if prediction==chords[label]: 
        correct+=1
    #if prediction is based on note name
    #if chords[prediction]==chords[label]:
    #    correct+=1
    #print(prediction,label,correct)
    

print("number correct: ",correct)

print("accuracy: ",correct/len(test))
                      

#run Decision Tree classifier
#clf=tree.DecisionTreeClassifier().fit(X_train,Y_train)
#export to Graphviz so that we can visualize the tree
if False:
    dot_data=tree.export_graphviz(clf,out_file=None,
                              #feature_names=relative,
                              #class_names=chromatic,
                              filled=True,rounded=True)
    graph=pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("dt_keys_relative.pdf")
#print(clf.score(X_test,Y_test))
'''
