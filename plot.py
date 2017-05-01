#plot distributions
import matplotlib.pyplot as plt

#diatonic values
majors=[0,14,16,5,7,21,23]
minors=[12,14,3,17,19,8,10]

with open("minor_average_dist.txt") as f:
    xnames,yvalues=[],[]
    for line in f:
        xnames.append(line.strip().split(",")[0])
        yvalues.append(float(line.strip().split(",")[1]))

xvalues=[i for i in range(24)]
barlist=plt.bar(xvalues,yvalues)
plt.title("Average Distribution of Chords for Minor Keys")
plt.xlabel("Relative Chord")
plt.ylabel("Percentage of Total")
#plt.legend(loc=1)
plt.xticks(xvalues,xnames)
#set colors for diatonics
for index in minors:
    barlist[index].set_color('r')
plt.show()
