# Machine Learning Engineer Nanodegree
## Capstone Project
Cody Farmer  
January 31st, 2018

## I. Definition
_(approx. 1-2 pages)_

### Project Overview
In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:
- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_

The goal of this project is to categorize tweets directed at the @AzureSupport handle into the closest Azure service category, such as "Web Applications". This falls into the general *machine learning* tasks called [*Classification*](https://en.wikipedia.org/wiki/Statistical_classification). The field of text classification is one of many *Natural Language Problems* that machine learning has been applied to effectively. After training the network, we will be able to predict the top three categories that an incoming tweet may belong.


 Our solution to this problem will utilize a *convolutional neural network* operating on word vectors created by a pre-trained *word2vec*. *word2vec* provides a method of turning words into arrays that represent various properties of each word. The end result will be the ability to take a tweet and provide the top three categories the network classifies the tweet's subject. 

### Problem Statement
In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_

Twitter has become a popular platform for users to seek help resolving technical issues for users. Although the platform doesn't serve well for troubleshooting, it's vital for any major brand to address these concerns. An integral part of this is sorting the issue into an appropriate category to allow for follow-up from people skilled in those services.

My project will take a single tweet and output the top three matching categories. This will be accomplished using a combination of two techniques, *word2vec* and *convolutional neural networks*. It's also important to properly encode the words before they are input into the model.

First, each tweet is sequenced using the Keras *tokenizer*. This is trained on all of the Tweets previously aimed at Azure. The purpose of the *tokenizer* is to turn words into numbers. It does this through a basic dictionary, resulting in a simple pattern like "Yes" equaling the number 1. Next, these sequences are fed into the *word2vec* network. We will use a pre-trained neural network that was trained by Google using an extremely large corpus (document collection) of text. This is important because classifying a specific set of tweets did not give us a lot of data in the scheme of things - and the pre-trained *word2vec* network helps us mitigate that. 

This *word2vec* network will output a vector representing each word fed into it. This vector attempts to capture the word's relationship to other words in the *corpus*. After feeding a Tweet into the model, we now have multiple vectors that represent each word. We take these vectors and combine them into a matrix - with each row of the matrix being the word vectors created. This will be operated on by a convolutional neural network to make predictions.

This trained model will then be used to predict the category of any given tweet, chosen by the highest probability categories. In order to make the prediction, the tweets will first need to be fed into the *tokenizer* and turned into sequences. A successful prediction will contain the correct category anywhere in the top three, but the training goal will be to make the correct prediction as the highest probability. 

### Metrics
Due to the uneven class distributions in our dataset, the choice of metric is important. Classical accuracy will result in training the network to be accurate only on the more common categories. For this reason, we will be utilizing the *log loss* function, which is in the *Keras* framework as *categorical_crossentropy*. Cross Entropy uses the probability of each category in the dataset to help create a more normalized accuracy metric.

## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

The dataset to train our neural network consists of 19,258 public tweets for the @AzureSupport twitter handle. These have each been categorized to one of 98 services by trained customer service agents working for Microsoft. The dataset has been provided to me in the CSV format. From this format, I will be tokenizing the tweets using the Keras tokenizer. This will allow me to utilize Gensim to import the word2vec pre-trained embedding weights and transform the tweets into a suitable word 100-dimension vector for training and testing. The dataset also contains other pieces of information, such as a sentiment estimation and followers. This dataset was obtained from Microsoft Support, as they have categorized these tweets already for record keeping. The frequency of the categories can be seen below, with some being extremely underrepresented, redundant, or overly specific. 

![Category Frequency in @AzureSupport Tweets](frequency.PNG "Frequency")

There are some challenges with learning out of this dataset. As it has been manually categorized and there isn't a clear way to define the right or wrong category, some of the data will not be the "correct" category. It's also not consistently labelled with more than one categories, where some tweets directly belong in multiple. This will make validation a little tricky, as the network may output a better categorization than the actual person. The data will be split with numpy into random sets to create training, validation, and testing sets - 80%, 10%, 10% respectively.

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

![Word Frequency for each category](treemap.PNG "Treemap")

The above plot is called a treemap. This treemap demonstrates the frequency of words per each category with the size of each box representing the volume of the word or category. In order to generate the dataset for this visualization, the Natural Language Tool Kit (NLTK) was used to filter out words by part of speech. The parts of speech eliminated to create this graph were pronouns, conjunctions, prepositions, pre-determiners, and interjections. Although these words may be useful for the algorithm's analysis of the Tweets, for visualization it provided the same top words for each category. This is due to some words being very common in questions, like "to" and "why". These words don't really help us visualize the differences in categories.

The bottom-right tan box represents the most common category - VMs. Unsurprisingly, you can see the most common word is "VM". The rest of the common words are still very similar between categories, even with removing the most obvious culprits. This chart demonstrates why this problem can't be solved through more simplistic methods such as word frequency. Some categories - such as billing questions represented in blue - shine in this graph with several different unique words as the most common tweeted. 



### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?
- _Are the techniques to be used thoroughly discussed and justified?
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?

The approach for this project, as mentioned above, is to use a Convolutional Neural network to classify tweets. In order to use a CNN, we first need to treat the data properly. These a few notable choices in this task.

**MAX_NB_WORDS** is the total allowed vocabulary size. I have chosen this to be 200,000 - a medium-sized number. The higher the number, the more accuracy you could theoretically have. However, at some point you are incurring a performance penalty for no accuracy gain.

**EMBEDDING_DIMENSIONS** is perhaps the most important constant in our entire program. This determines the length of vectors used to represent our words as word embeddings. Much like the previous constant, the larger the better. You can think of each column in an embedding vector as a feature of the word being vectorized. However, the curse of dimensionality is in effect here. The size of the embedding matrix for us will be [EMBEDDING\_DIMENSIONS, MAX\_NB\_WORDS]. 

**MAX_SEQUENCE_LENGTH** is the maximum length of a sequence that will be output from the input tweet. A _sequence_ is just a numerical representation of words. In our case, we're turning each word of a tweet into a number in our sequence. We've chosen the maximum length to be 50 words. This should encompass almost all tweets, as that would be an average word length of 6 letters to hit the tweet character limit. 

**MIN_SAMPLES** defines the minimum number of samples a category needs to contain in order to train/test on it. This is set to 50 for our purposes. A higher requirement for samples will lead to better accuracy, but can be difficult to obtain. 

These constants are used to prepare the Tweets for processing through the CNN. This CNN is will work on a matrix of the Tweet, instead of the more common method of operating on averages or sums of the _word2vec_ embeddings. As each sequence is fed into the CNN, filters attempt to pick out the most relevant patterns present in the input data. On top of these filters, we create pools to allow the network to detect the pattern in differnt positions within the tweet. This process is the basis for all CNNs. In our network, the _word2vec_ embeddings are fed into four different parallel filters. Each set of filters is individually pooled, then concatenated together. This total array is fed into a dense neural network that is globally averaged together. Then we've created another dense network with a _softmax_ activation. This outputs the likelyhood of each tweet belong to a particular category. 




### Benchmark

In order to establish the effectiveness of my model, I consider two metrics: _Categorical Accuracy_ and _Top 3 Accuracy_. We have available a very good paper to compare against in Yoon Kim's ^[1] work. Our model corresponds closely with what she defines as CNN-static. Her model's accuracy over several different test sets ranges from 81.5% to 93.4% (ignoring SST-1, which I consider an outlier). We will consider our model a success if the _Categorical Accuracy_ surpasses the minimum of those tests - 81.5%.



## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

Our data is provided as large CSV files with classification performed by hand. Along with the category, there are several other fields of information for each Tweet we will train on. These columns aren't of value for us, so I immediately drop them from the dataframe. In addition to removing un-wanted information, Tweets that aren't categorized need to be removed. This can be accomplished by converting the empty fields to NAN values and using the fast dropna method. This accomplished the basic cleaning needed for the data. Next, we perform some actions a little more specific to our particular algorithm.

In order to properly train for a category, we will need enough samples to allow the network to learn patterns. This value is mostly based on experience, and I've set it to 50 as my baseline. The constant can be adjusted, allowing for further tuning. Addtionally, neural networks work best with their targets expressed as integers. We'll accomplish this by _coding_ the categories, using the built-in Pandas method. After coding, each category will have a single corresponding number. For example, 'Active Directory' will be '3'.

Similar to encoding categories to numbers, the input tweets themselves need a representation that is easily understood by the neural networks. This is accomplished by using a _Tokenizer_. Once fit onto a corpus, the _Tokenizer_ will assign a number to each word. This allows us to turn any text into an array of numbers that can be processed by the neural network. Conveniently, Keras included a _Tokenizer_. Our implementation will utilize this Keras function.



### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

In order to perform this classification, I implemented a two-part neural network in Keras. The solution was created and ran inside of a Jupyter Notebook utilizing Tensorflow-GPU. 

The first neural network is a _word2vec_ network that was pre-trained on an extremely large corpus (a set of text). The vectors created from a word2vec network are called _Embeddings_. They are a popular way to represent textual information within machine learning. The created embeddings contain a representation for the word that is learned from the corpus, which can reflect the relationship to other words. In my network, I locked the learned embeddings from being trained. In this case, I'm relying completely on the Google-trained set "GoogleNews-Vectors-negative300". The 300 here is an important value, as it represents the length of each embedding vector.



### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?

[1] Kim, Yoon https://arxiv.org/pdf/1408.5882.pdf