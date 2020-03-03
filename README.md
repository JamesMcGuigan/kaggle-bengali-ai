# Kaggle Competition Entry: Bengali AI

https://www.kaggle.com/c/bengaliai-cv19

Bengali is the 5th most spoken language in the world with hundreds of million of speakers. It’s the official
language of Bangladesh and the second most spoken language in India. Considering its reach, there’s significant
business and educational interest in developing AI that can optically recognize images of the language handwritten
. This challenge hopes to improve on approaches to Bengali recognition.

Optical character recognition is particularly challenging for Bengali. While Bengali has 49 letters (to be more
specific 11 vowels and 38 consonants) in its alphabet, there are also 18 potential diacritics, or accents. This
means that there are many more graphemes, or the smallest units in a written language. The added complexity results
in ~13,000 different grapheme variations (compared to English’s 250 graphemic units).

For this competition, you’re given the image of a handwritten Bengali grapheme and are challenged to separately
classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics.

---
## Leaderboard
Deadline: 2020-03-16

| Position       | Score   |  Prize       |
|----------------|:-------:|-------------:|
| 1st Place      |  0.9927 | $5000        |
| 2nd Place      |  0.9907 | $2000        |
| 3rd Place      |  0.9907 | $1000        |
| Top 13         |  0.9901 | Gold Medal   |
| Top 5%  (89)   |  0.9803 | Silver Medal |
| Top 10% (179)  |  0.9721 | Bronze Medal | 


## Dataset
```
kaggle competitions download -c bengaliai-cv19 -p ./data/
unzip ./data/bengaliai-cv19.zip -d ./data/
```

This dataset contains images of individual hand-written [Bengali characters](https://en.wikipedia.org/wiki/Bengali_alphabet). Bengali characters (graphemes) are written by combining three components: a grapheme_root
, vowel_diacritic, and consonant_diacritic. Your challenge is to classify the components of the grapheme in each
image. There are roughly 10,000 possible graphemes, of which roughly 1,000 are represented in the training set. The
test set includes some graphemes that do not exist in train but has no new grapheme components. It takes a lot of
volunteers filling out [sheets like this](https://github.com/BengaliAI/graphemePrepare/blob/master/collection/A4/form_1.jpg)
to generate a useful amount of real data; focusing the problem on the grapheme components rather than on recognizing
whole graphemes should make it possible to assemble a Bengali OCR system without handwriting samples for all 10,000
graphemes.

**data/train.csv**
```
image_id,grapheme_root,vowel_diacritic,consonant_diacritic,grapheme
Train_0,15,9,5,ক্ট্রো
Train_1,159,0,0,হ
Train_2,22,3,5,খ্রী
```

**data/test.csv**
```
row_id,image_id,component
Test_0_consonant_diacritic,Test_0,consonant_diacritic
Test_0_grapheme_root,Test_0,grapheme_root
Test_0_vowel_diacritic,Test_0,vowel_diacritic
```

**data/sample_submission.csv**
```
row_id,target
Test_0_consonant_diacritic,0
Test_0_grapheme_root,0
Test_0_vowel_diacritic,0
```