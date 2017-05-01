#read in metadata and match songs from msd

def read_metadata(filename="new_metadata.csv"):
    songs={}
    with open(filename) as f:
        for line in f:
            artist=convert_name(line.split(",")[3])
            song=convert_name(line.split(",")[4])
            songs.setdefault(artist,{})
            #save song id for use later
            songs[artist][song]=line.split(",")[1]
    return songs


def read_songs(filename="msd_keys.txt"):
    songs={}
    notes=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    modes=['m','']
    songid=None
    with open(filename,encoding='latin-1',errors="replace") as f:
        try:
            for line in f:
                line=line.strip().split(",")
                songid=line[0]
                artist=convert_name(line[4].split(": ")[1])
                song=convert_name(line[5].split(": ")[1])
                key=notes[int(line[3].split(": ")[1])]
                key+=modes[int(line[2].split(": ")[1])]
                songs.setdefault(artist,{})
                songs[artist][song]=key
        except:
            print(songid)
    return songs


def convert_name(name):
    #convert name into standard format
    stop_chars=['"',"'",",",".","?","!","(",")"," ","[","]"]
    name=name.lower()
    for c in stop_chars: 
        name=name.replace(c,"")
    name=name.replace('&','and').replace('featuring','ft')
    return name

def compareDatasets(data1,data2):
    #assumes two dictionaries
    matches=set([]) #unique set of (artist,song) tuples
    for artist in data1:
        if artist in data2:
            for song in data1[artist]:
                if song in data2[artist]:
                    matches.add((artist,song))
    return matches

def writeMatches(matches,songs,md):
    #write to file; include key from songs
    with open('matches.txt','w') as f:
        for tup in matches:
            line=list(tup)
            line.append(songs[tup[0]][tup[1]]) #add key
            line.append(md[tup[0]][tup[1]]) #add songid from Ultimate Guitar
            f.write(",".join(line)+'\n')

#code we ran
#md=read_metadata()
#songs=read_songs()
#matches=compareDatasets(md,songs)
#print(len(matches))
#writeMatches(matches,songs,md)
