################# packages needed
import pandas as pd
import pickle
import collections
from operator import itemgetter
import random
import classify_kb as classify
#################

def subfinder(mylist, pattern): 
    ''' given a list and pattern, the function finds the locations of the patterns'''

    matches = [] # initialize list
    length = len(pattern) # stores length of patterns
    for i in range(len(mylist)): # for each value
        #if list is long enough, first value is correct, and rest is correct
        if i+length < len(mylist) and mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:                      
            #append the location of the new chord
            matches.append(i+length)
    return matches

def suggestions_finder(chords, possible_key):
    '''Given the chords and the possible keys, the function finds the suggestions
    1. similar suggestions: find songs in the key, find songs with the pattern in it, suggest the next chord
    2. creative suggestions: take all other songs, convert the chords to the chromatic scale, find the pattern
        in the songs with the same relative difference, append next chord'''

    #similar
    similar_song_ids = matches[matches.key.isin(possible_key)].songID.tolist() # get list of similar song ids
    similar_idxs = [i for i in similar_song_ids if chords in data_chromatic[i]] # finds indexes in which user input pattern appears

    # look through each song with pattern, split song by pattern, get next chord
    similar_suggestions = [int(values.split('-')[1]) for ids in similar_idxs for values in data_chromatic[ids].split(chords)[1:] if values != '-' and len(values) > 0]

    #creative relative
    creative_keys = list(set(chromatic.keys()) - set(possible_key))
    creative_song_ids = matches[matches.key.isin(creative_keys)].songID.tolist() # get list of creative song ids
    split_chords = chords.split('-') # split chords 
    ## finds difference, converts to str, combines to str sep by comma
    split_chords_difference = ',' + ','.join([str(int(j)-int(i)) for i, j in zip(split_chords[:-1], split_chords[1:])]) 
    creative_idxs = [i for i in creative_song_ids if split_chords_difference in data_difference_chromatic[i]]
    split_chords_difference = split_chords_difference.split(',')[1:]
    # list of list of suggestions by finding the location of the pattern and 
    creative_suggestions = [subfinder(data_difference_chromatic[x].split(','),split_chords_difference) for x in creative_idxs]
    # find the corresponding value in the chords
    creative_suggestions = [list(map(lambda x: data_chromatic[creative_idxs[y]].split('-')[x], creative_suggestions[y])) for y in range(len(creative_idxs))]
    # remove empty lists -> in case there is no next value
    creative_suggestions = [x for x in creative_suggestions if len(x) > 0]
    # flatten lists
    creative_suggestions = [int(x) for sub in creative_suggestions for x in sub]
    
    return similar_suggestions, creative_suggestions

def check_valid(chord):
    ''' check if chords are valid
    if size 1: check if 1st is [A-G]
    if size 2: also checks if 2nd is flat, sharp, minor
    if size 3: also checks if 3d is minor'''
    return ((len(chord) == 1 and chord[0] >= 'A' and chord[0] <= 'G') or \
        (len(chord) == 2 and chord[0] >= 'A' and chord[0] <= 'G' and chord[1] in ['b','#','m']) or \
        (len(chord) == 3 and chord[0] >= 'A' and chord[0] <= 'G' and chord[1] in ['b','#','m'] and chord[2] == 'm')) 

chromatic = { # chromatic scale conversion from notes to value
    'C': 0,'C#': 1,'Cb': 11,
    'D':2,'D#':3,'Db':1,
    'E':4,'E#':5,'Eb':3,
    'F':5,'F#':6,'Fb':4,
    'G': 7,'G#':8,'Gb':6,
    'A':9,'A#':10,'Ab':8,
    'B': 11,'B#':0,'Bb':10,
    
    'Cm': 12,'C#m': 2,'Cbm': 23,    
    'Dm':14,'D#m':15,'Dbm':13,
    'Em':16,'E#m':17,'Ebm':15,
    'Fm':17,'F#m':18,'Fbm':16,
    'Gm': 19,'G#m':20,'Gbm':18,
    'Am':21,'A#m':22,'Abm':20,
    'Bm': 23,'B#m':12,'Bbm':22,
}

reverse_chromatic = { # chromatic scale conversion from values to notes
     0: ['C', 'B#'],
     1: ['C#', 'Db'],
     2: ['D', 'C#m'],
     3: ['D#', 'Eb'],
     4: ['E', 'Fb'],
     5: ['E#', 'F'],
     6: ['F#', 'Gb'],
     7: ['G'],
     8: ['G#', 'Ab'],
     9: ['A'],
     10: ['A#', 'Bb'],
     11: ['Cb', 'B'],
     12: ['Cm', 'B#m'],
     13: ['Dbm'],
     14: ['Dm'],
     15: ['D#m', 'Ebm'],
     16: ['Em', 'Fbm'],
     17: ['E#m', 'Fm'],
     18: ['F#m', 'Gbm'],
     19: ['Gm'],
     20: ['G#m', 'Abm'],
     21: ['Am'],
     22: ['A#m', 'Bbm'],
     23: ['Cbm', 'Bm']}

keys = { # scales for each key
    'C':[ 'C', 'D', 'E', 'F', 'G', 'A','B'],
    'Cm':['C', 'D', 'Eb', 'F', 'G', 'Ab','Bb'],
    'C#':['C#', 'D#', 'E#', 'F#', 'G#', 'A#', 'B#'],
    'C#m':['C#', 'D#', 'E', 'F#', 'G#', 'A','B'],
    'Cb':[ 'Cb', 'Db', 'Eb', 'Fb', 'Gb', 'Ab', 'Bb'],
    'Cbm':[ 'Cb', 'Db', 'Ebb', 'Fb', 'Gb', 'Abb', 'Bbb'],
    
    'D':[ 'D', 'E','F#', 'G', 'A', 'B', 'C#'],
    'Dm':['D', 'E', 'F', 'G', 'A', 'Bb', 'C'],
    'D#':['D#', 'E#', 'F##', 'G#', 'A#', 'B#', 'C##'],
    'D#m':['D#', 'E#', 'F#', 'G#', 'A#', 'B', 'C#'],
    'Db':['Db', 'Eb', 'F', 'Gb', 'Ab', 'Bb', 'C'],
    'Dbm':['Db', 'Eb', 'Fb', 'Gb', 'Ab', 'Bbb', 'Cb'],
    
    'E':['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'],
    'Em':['E', 'F#', 'G', 'A', 'B', 'C','D'],
    'E#':['E#', 'F##', 'G##', 'A#', 'B#', 'C##', 'D##'],
    'E#m':['E#', 'F##', 'G#', 'A#', 'B#', 'C#', 'D#'],
    'Eb':['Eb', 'F', 'G', 'Ab', 'Bb', 'C', 'D'],
    'Ebm':['Eb', 'F', 'Gb', 'Ab', 'Bb', 'Cb', 'Db'],
    
    'F':['F', 'G', 'A', 'Bb', 'C', 'D', 'E'],
    'Fm':['F', 'G', 'Ab', 'Bb', 'C', 'Db','Eb'],
    'F#':['F#', 'G#', 'A#', 'B', 'C#', 'D#', 'E#'],
    'F#m':['F#', 'G#', 'A', 'B', 'C#', 'D','E'],
    'Fb':['Fb', 'Gb', 'Ab', 'Bbb', 'Cb', 'Db', 'Eb'],
    'Fbm':['Fb', 'Gb', 'Abb', 'Bbb', 'Cb', 'Dbb', 'Ebb'],
    
    'G':['G', 'A', 'B', 'C', 'D', 'E','F#'],
    'Gm':[ 'G', 'A', 'Bb', 'C', 'D', 'Eb', 'F'],
    'G#':['G#', 'A#', 'B#', 'C#', 'D#', 'E#', 'F##'],
    'G#m':['G#', 'A#', 'B', 'C#', 'D#', 'E', 'F#'],
    'Gb':['Gb', 'Ab', 'Bb', 'Cb', 'Db', 'Eb', 'F'],
    'Gbm':['Gb', 'Ab', 'Bbb', 'Cb', 'Db', 'Ebb', 'Fb'],
    
    'A':['A', 'B', 'C#', 'D', 'E', 'F#','G#'],
    'Am':['A', 'B', 'C', 'D', 'E', 'F','G'],
    'A#':[ 'A#', 'B#', 'C##', 'D#', 'E#', 'F##', 'G#'],
    'A#m':[ 'A#', 'B#', 'C#', 'D#', 'E#', 'F#', 'G#'],
    'Ab':['Ab', 'Bb', 'C', 'Db', 'Eb', 'F','G'],
    'Abm':[ 'Ab', 'Bb', 'Cb', 'Db', 'Eb', 'Fb','Gb'],
    
    'B':['B', 'C#', 'D#', 'E', 'F#', 'G#','A#'],
    'Bm':['B', 'C#', 'D', 'E', 'F#', 'G','A'],
    'B#':['B#', 'C##', 'D##', 'E#', 'F##', 'G##','A##'],
    'B#m':['B#', 'C##', 'D#', 'E#', 'F##', 'G#','A#'],
    'Bb':[ 'Bb', 'C', 'D', 'Eb', 'F', 'G', 'A'],
    'Bbm':[ 'Bb', 'C', 'Db', 'Eb', 'F', 'Gb', 'Ab'], 
}

################################################## reading,transforming data
matches = pd.read_csv("matches.txt", sep = ',', names = ['artist', 'song', 'key', 'songID']) # songID and key
data = pickle.load( open( "new_data.p", "rb" ) ) ## song chord data
data_chromatic = {x:'-'.join([str(chromatic[a]) for a in y]) for x,y in data.items()} # converts data into string of numbers
## finding difference between chromatic chords
data_difference_chromatic = {x:','+','.join([str(int(j)-int(i)) for i, j in zip(y.split('-')[:-1], y.split('-')[1:])]) for x,y in data_chromatic.items()}
##################################################

### printing info so that user knows how to input
print("--------------------------------")
print("Root note is capitalized [A-G]")
print("b = flat, # = sharp")
print("m = minor, default Major") 
print("Type \'exit\' to quit.") 
print("--------------------------------")
###

################################################## gets, cleans chords from user
user_input = (input("Enter chords separated by comma: ")).replace(' ','').split(',') # ask user for list of chords
if type(user_input) == str and user_input.lower() == 'exit': exit() # exit() requested by user

old_user_input = user_input.copy() # make deep copy of input 
print('chords = ', user_input) # print for user to see
del_list = [] # initialize a list of chords to delete
for pos in range(len(user_input)): # for each index
    chord = user_input[pos]  # save the chord value
    while True: # always run
        if check_valid(chord): # if its not a valid chord  
            break # get out of while loop if the chord is valid
        print('Chord at position', pos, 'is not valid') # print the chord is invalid 
        checking = input('Would you like to Replace or Delete? (R/D) ').lower()
        if type(checking) == str and checking.lower() == 'exit': exit()
        if checking == 'd': # if user wants to delete
            del_list.append(pos) # add index to delete list
            break # break out of while loop
        elif checking == 'r': # if user wants to replace
            chord = input('What chord would you like to add instead? (<key>) ') # ask input for new chord
            if type(chord) == str and chord.lower() == 'exit': exit()
            user_input[pos] = chord # save the new chord into the list
        else: print("Not valid input. Try again ...")

user_input = [user_input[x] for x in range(len(user_input)) if x not in del_list] # remove chords the user wants to delete
if user_input != old_user_input: print('chords = ',user_input) # print chords
################################################## 

def readClassifiers():
    '''return classifiers for absolute distribution and major/minor relative distributions'''
    with open('svm_major.pkl','rb') as f:
        clf_major=pickle.load(f)
    with open('svm_minor.pkl','rb') as f:
        clf_minor=pickle.load(f)
    with open('svm_absolute.pkl','rb') as f:
        clf_abs=pickle.load(f)
    return (clf_major,clf_minor,clf_abs)

def run(user_input):
    '''given chords from user, the function finds the list of possible keys, it then asks the user to verify each one and suggest some of their own.
    It also asks the user for a creativity factor. It then gathers the suggestions and presents them to the user. If the user doesn't want the chord,
    it presents the next suggestion. If there are no suggestions left, it returns a string to print. If the user wants the chord, it returns a list of
    the user inputted chords with the new chord.'''


    ######################################### machine learning to get list of keys
    #########################################
    #########################################
    #########################################
    #########################################
    clf_major,clf_minor,clf_abs=readClassifiers()
    #try all chromatic keys
    test_keys=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','Cm','C#m','Dm','D#m','Em','Fm','F#m','Gm','G#m','Am','A#m','Bm']
    #returns sorted list of relative chromatic scores
    scores=classify.test_chords(test_keys,user_input,chromatic,clf_major,clf_minor,first=False,last=False)
    #returns sorted list of absolute chromatic scores
    abs_scores=classify.test_song(user_input,chromatic,clf_abs,test_keys,first=False,last=False)
    #combine scores together for predictions
    final_scores=[]
    for chord1 in scores:
        for chord2 in abs_scores:
            if chromatic[chord1[1]]==chromatic[chord2[1]]:
                final_scores.append((chord1[0]*chord2[0],chord1[1])) #choose first representation by default
    total=sum([score[0] for score in final_scores])
    #turn final scores into readable format and normalize
    final_scores=[(chord[1],chord[0]/total) for chord in sorted(final_scores,reverse=True)]
    #print(scores)
    #print(abs_scores)
    print('\n'.join(tup[0]+", Confidence Score: "+str(tup[1]) for tup in final_scores[:5] if tup[1]>0))


    possible_key = [key[0] for key in final_scores[:5]]

    # user feedback on possible keys
    print("--------------------------------\nSome possible keys were found.")
    possible_key = [x if input("Keep " + x + '?: (Y/N) ').lower() == 'y' else exit() for x in possible_key] # ask which keys the user wants to keep
    if input("Would you like to add any? (Y/N) ").lower() == 'y': # if user wants to add some
        key = input('What key would you like to add? (<key>/<blank>) ') # ask user for key or blank
        if type(key) == str and key.lower() == 'exit': exit() # exit() requested by user
        while key: # if key given
            if check_valid(key):possible_key.append(key) # if valid key, append to possible key list
            else: print(str(key) + ' is not a valid key') # if not valid, say its not valid
            key = input('What key would you like to add? (<key>/<blank>) ') # ask user again
            if type(key) == str and key.lower() == 'exit': exit() # exit() requested by user
            possible_key.append(key) # append the key to the list

    # get creative values
    creativity = None # set default value
    while type(creativity) != float or creativity < 0 or creativity > 1: # while the inputted ratio is not correct
        creativity = input('How creative do you want the suggestions to be? decimal from [0,1] ') # ask the user
        if type(creativity) == str and creativity.lower() == 'exit': exit() # exit() requested by user
        try: # try floating it
            creativity = float(creativity)
        except ValueError: # if it is not a number
            print('Not a number. Try again ...') # print error
            continue # continue to next iteration
        if creativity < 0 or creativity > 1: # if the number is not correct
            print('Not a valid number. Try again ...') # print error
            continue # continue to next iteration


    chromatic_user_input = '-'.join(str(chromatic[x]) for x in user_input) # convert user input to string of numbers
    similar_suggestions,creative_suggestions = suggestions_finder(chromatic_user_input, possible_key) # find suggestions

    # if no cases available: drop the first chord from trial and try again
    while not(creative_suggestions or similar_suggestions): # if both suggestions are empty
        print("No cases available. Trying without first chord ...")
        chromatic_user_input = '-'.join(chromatic_user_input.split('-')[1:]) # remove first chord and try again
        similar_suggestions,creative_suggestions = suggestions_finder(chromatic_user_input, possible_key) # run again

    similar = collections.Counter(similar_suggestions) # get count dict of similar suggestions
    creative = collections.Counter(creative_suggestions) # get count dict of creative suggestions
    creative = collections.Counter({x:creativity*y for x,y in creative.items()}) # factor count dict of creative suggestions by creativity
    total_suggestions = creative+similar # sum count dicts


    # return suggestion by finding the most common
    # convert chromatic to chord
    # if multiple chords, check which one appers most frequently in the possible keys
    notes = [note for key in possible_key for note in keys[key]] # make a list of all the notes in the keys
    for suggestion in total_suggestions.most_common(): # start with most commonly occuring value
        suggestion = suggestion[0] # get chord number
        chromatic_value = reverse_chromatic[suggestion] # get possible list of chords
        freq_counts = [(x,notes.count(x)) for x in chromatic_value] # list of tuples of chord and count
        random.shuffle(freq_counts) # shuffle in case there is a tie
        value = max(freq_counts,key=itemgetter(1))[0] # choose the chord that appears the most
        response = input('We suggest ' + value + '. Do you want to add it to the list of chords? [Y/N] ').lower() # ask user if they want the chord
        if type(response) == str and response.lower() == 'exit': exit() # exit() if user wants
        if response == 'n': # if user doesn't want
            continue # continue
        if response == 'y': # if user wants the chord
            return user_input + [value] # append new chord to user inputted chords

    return 'No more suggestions left.' # return string of no suggestions available



while True: ## always run
    response = run(user_input) # get response from run function
    if type(response) == str: # if its a string
        print(response) # print the response
        break # break out of the cycle
    elif type(response) == list: # if its a list
        print('chords = ',response) # print the new set of chords
        response = run(response) # run function again

