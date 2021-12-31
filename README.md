# Kaggle-Jigsaw-Rate-Severity-of-Toxic-Comments

## **Goal**
To rank the relative ratings of toxicity between comments from the Wikipedia Talk page

## **Outcome**
Reached 0.802/1.000 public score on Kaggle

## **Dataset**
### Training data:
 
|id|comment_text|toxic|severe_toxic|obscene|threat|insult|identity_hate|
|---|---|---|---|---|---|---|---|
|0000997932d777bf|Explanation\nWhy the edits made under my usern..|    0|1|0|0|0|0|
|000103f0d9cfb60f|D'aww! He matches this background colour I'm s..|    0|0|0|1|1|0|
|000113f07ec002fd|Hey man, I'm really not trying to edit war. It..|    1|0|0|0|1|1|
|0001b41b1c6bb37e|"\nMore\nI can't make any real suggestions on ..|    0|1|1|0|0|0|
|0001d958c54c6e35|You, sir, are my hero. Any chance you remember...|0|0|0|1|0|0|

**id** : unique id of each comment  
**comment_text** : comment text crawled from Wikipedia Talk page  
**toxic**, **severe_toxic**, **obscene**, **obscene**, **threat**, **insult**, **identity_hate** : labels generated from previous competition that match the column description 

### Validation data:  
Pair of comments where experts have marked one to be more toxic than the other.

|worker|less_toxic|more_toxic|
|---|---|---|
|313|This article sucks \n\nwoo woo wooooooo|WHAT!!!!!!!!?!?!!?!?!!?!?!?!?!!!!!!!!!!!!!!!!!...|
|188|"And yes, people should recognize that but the...|Daphne Guinness \n\nTop of the mornin' my fav...|
|82|Western Media?\n\nYup, because every crime in...|"Atom you don't believe actual photos of mastu...|
|347|And you removed it! You numbskull! I don't car...|You seem to have sand in your vagina.\n\nMight...|
|539|smelly vagina \n\nBluerasberry why don't you ...|hey \n\nway to support nazis, you racist|


### Testing data:
Raw text to be predicted the toxicity ranking

|comment_id|text|
|---|---|
|114890|"\n \n\nGjalexei, you asked about whether ther...|
|732895|Looks like be have an abuser , can you please ...|
|1139051|I confess to having complete (and apparently b...|
|1434512|"\n\nFreud's ideas are certainly much discusse...|
|2084821|It is not just you. This is a laundry list of ...|

## Method
- Data Pre-processing:
  - Calculate weightings
    - We want to generate a singular **y** value for use in a regression model, so we give weights to each tag column and sum up their values to create a final *"level"* field.
    - Weights were chosen after trial and error, with the final weights being obtained from a trained model of a previous competition.
  - Tokenization
  - Lemmatization
  - Data Cleaning
    - Remove internal Wiki links, signatures and time stamps
    - Remove URLs
    - Replace common profanity filter evasion words with their corresponding words
    - Replace repetitive characters 
  
- Adding Features:
  - Punctuation percentage
  - Uppercase percentage
  - Average sentence length
  - Profanity Words percentage
  - Total comment length
- Encoding:
    - Tf-Idf Vectorizer + generated features
- 


## Problems Encountered
(1) Score of Validation data 

## How to Solve them


Kaggle Competition Overview:
https://www.kaggle.com/c/jigsaw-toxic-severity-rating/overview