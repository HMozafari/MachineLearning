import pandas as pd
import numpy as np

newdata = pd.read_csv('annotations_final.csv', sep='\t')
print(newdata.head(n=5))

print (newdata.info())
print(newdata.columns)

print(newdata[["clip_id", "mp3_path"]])

# Previous command extracted it as a Dataframe. We need it as a matrix to do analyics on.
# Extract clip_id and mp3_path as a matrix.
clip_id, mp3_path = newdata[["clip_id", "mp3_path"]].as_matrix()[:,0], newdata[["clip_id", "mp3_path"]].as_matrix()[:,1]


synonyms = [['beat', 'beats'],
            ['chant', 'chanting'],
            ['choir', 'choral'],
            ['classical', 'clasical', 'classic'],
            ['drum', 'drums'],
            ['electro', 'electronic', 'electronica', 'electric'],
            ['fast', 'fast beat', 'quick'],
            ['female', 'female singer', 'female singing', 'female vocals', 'female vocal', 'female voice', 'woman', 'woman singing', 'women'],
            ['flute', 'flutes'],
            ['guitar', 'guitars'],
            ['hard', 'hard rock'],
            ['harpsichord', 'harpsicord'],
            ['heavy', 'heavy metal', 'metal'],
            ['horn', 'horns'],
            ['india', 'indian'],
            ['jazz', 'jazzy'],
            ['male', 'male singer', 'male vocal', 'male vocals', 'male voice', 'man', 'man singing', 'men'],
            ['no beat', 'no drums'],
            ['no singer', 'no singing', 'no vocal','no vocals', 'no voice', 'no voices', 'instrumental'],
            ['opera', 'operatic'],
            ['orchestra', 'orchestral'],
            ['quiet', 'silence'],
            ['singer', 'singing'],
            ['space', 'spacey'],
            ['string', 'strings'],
            ['synth', 'synthesizer'],
            ['violin', 'violins'],
            ['vocal', 'vocals', 'voice', 'voices'],
            ['strange', 'weird']]


# Merge the synonyms and drop all other columns than the first one.

# Example:
# Merge 'beat', 'beats' and save it to 'beat'.
# Merge 'classical', 'clasical', 'classic' and save it to 'classical'.

for synonym_list in synonyms:
    newdata[synonym_list[0]] = newdata[synonym_list].max(axis=1)
    newdata.drop(synonym_list[1:], axis=1, inplace=True)

newdata.info()


# Drop the mp3_path tag from the dataframe
newdata.drop('mp3_path', axis=1, inplace=True)
# Save the column names into a variable
data = newdata.sum(axis=0)

data.sort_values(axis=0, inplace=True)


# Find the top tags from the dataframe.
topindex, topvalues = list(data.index[84:]), data.values[84:]
del(topindex[-1])
topvalues = np.delete(topvalues, -1)

print(topindex)

rem_cols = data.index[:84]
