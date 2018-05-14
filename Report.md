# Machine Learning Engineer Nanodegree
## Capstone Project
Cody Farmer  
January 31st, 2018

## I. Definition
_(approx. 1-2 pages)_

### Project Overview

The goal of this project is to categorize tweets directed at the @AzureSupport handle into the closest Azure service category, such as "Web Applications". This falls into the general *machine learning* tasks called [*Classification*](https://en.wikipedia.org/wiki/Statistical_classification). The field of text classification is one of many *Natural Language Problems* that machine learning has been applied to effectively. After training the network, we will be able to predict the top three categories that an incoming tweet may belong.


 Our solution to this problem will utilize a *convolutional neural network* operating on word vectors created by a pre-trained *word2vec*. *word2vec* provides a method of turning words into arrays that represent various properties of each word. The end result will be the ability to take a tweet and provide the top three categories the network classifies the tweet's subject. 

### Problem Statement

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

The dataset to train our neural network consists of 19,258 public tweets for the @AzureSupport twitter handle. These have each been categorized to one of 98 services by trained customer service agents working for Microsoft. The dataset has been provided to me in the CSV format. From this format, I will be tokenizing the tweets using the Keras tokenizer. This will allow me to utilize Gensim to import the word2vec pre-trained embedding weights and transform the tweets into a suitable word 100-dimension vector for training and testing. The dataset also contains other pieces of information, such as a sentiment estimation and followers. This dataset was obtained from Microsoft Support, as they have categorized these tweets already for record keeping. The frequency of the categories can be seen below, with some being extremely underrepresented, redundant, or overly specific. 

![Category Frequency in @AzureSupport Tweets](frequency.PNG "Frequency")

There are some challenges with learning out of this dataset. As it has been manually categorized and there isn't a clear way to define the right or wrong category, some of the data will not be the "correct" category. It's also not consistently labelled with more than one categories, where some tweets directly belong in multiple. This will make validation a little tricky, as the network may output a better categorization than the actual person. The data will be split with numpy into random sets to create training, validation, and testing sets - 80%, 10%, 10% respectively.

### Exploratory Visualization

![Word Frequency for each category](treemap.PNG "Treemap")

The above plot is called a treemap. This treemap demonstrates the frequency of words per each category with the size of each box representing the volume of the word or category. In order to generate the dataset for this visualization, the Natural Language Tool Kit (NLTK) was used to filter out words by part of speech. The parts of speech eliminated to create this graph were pronouns, conjunctions, prepositions, pre-determiners, and interjections. Although these words may be useful for the algorithm's analysis of the Tweets, for visualization it provided the same top words for each category. This is due to some words being very common in questions, like "to" and "why". These words don't really help us visualize the differences in categories.

The bottom-right tan box represents the most common category - VMs. Unsurprisingly, you can see the most common word is "VM". The rest of the common words are still very similar between categories, even with removing the most obvious culprits. This chart demonstrates why this problem can't be solved through more simplistic methods such as word frequency. Some categories - such as billing questions represented in blue - shine in this graph with several different unique words as the most common tweeted. 



### Algorithms and Techniques

The approach for this project, as mentioned above, is to use a Convolutional Neural network to classify tweets. In order to use a CNN, we first need to treat the data properly. These a few notable choices in this task.

**MAX_NB_WORDS** is the total allowed vocabulary size. I have chosen this to be 200,000 - a medium-sized number. The higher the number, the more accuracy you could theoretically have. However, at some point you are incurring a performance penalty for no accuracy gain.

**EMBEDDING_DIMENSIONS** is perhaps the most important constant in our entire program. This determines the length of vectors used to represent our words as word embeddings. Much like the previous constant, the larger the better. You can think of each column in an embedding vector as a feature of the word being vectorized. However, the curse of dimensionality is in effect here. The size of the embedding matrix for us will be [EMBEDDING\_DIMENSIONS, MAX\_NB\_WORDS]. 

**MAX_SEQUENCE_LENGTH** is the maximum length of a sequence that will be output from the input tweet. A _sequence_ is just a numerical representation of words. In our case, we're turning each word of a tweet into a number in our sequence. We've chosen the maximum length to be 50 words. This should encompass almost all tweets, as that would be an average word length of 6 letters to hit the tweet character limit. 

**MIN_SAMPLES** defines the minimum number of samples a category needs to contain in order to train/test on it. This is set to 50 for our purposes. A higher requirement for samples will lead to better accuracy, but can be difficult to obtain. 

These constants are used to prepare the Tweets for processing through the CNN. This CNN is will work on a matrix of the Tweet, instead of the more common method of operating on averages or sums of the _word2vec_ embeddings. As each sequence is fed into the CNN, filters attempt to pick out the most relevant patterns present in the input data. On top of these filters, we create pools to allow the network to detect the pattern in differnt positions within the tweet. This process is the basis for all CNNs. In our network, the _word2vec_ embeddings are fed into four different parallel filters. Each set of filters is individually pooled, then concatenated together. This total array is fed into a dense neural network that is globally averaged together. Then we've created another dense network with a _softmax_ activation. This outputs the likelyhood of each tweet belong to a particular category. 




### Benchmark

In order to establish the effectiveness of my model, I consider two metrics: _Categorical Accuracy_ and _Top 3 Accuracy_. We have available a very good paper to compare against in Yoon Kim's ^[1] work. Our model corresponds closely with what she defines as CNN-static. Her model's accuracy over several different test sets ranges from 81.5% to 93.4% (ignoring SST-1, which I consider an outlier). We will consider our model a success if the _Categorical Accuracy_ surpasses the minimum of those tests - 81.5%.



## III. Methodology

### Data Preprocessing


Our data is provided as large CSV files with classification performed by hand. Along with the category, there are several other fields of information for each Tweet we will train on. These columns aren't of value for us, so I immediately drop them from the dataframe. In addition to removing un-wanted information, Tweets that aren't categorized need to be removed. This can be accomplished by converting the empty fields to NAN values and using the fast dropna method. This accomplished the basic cleaning needed for the data. Next, we perform some actions a little more specific to our particular algorithm.

In order to properly train for a category, we will need enough samples to allow the network to learn patterns. This value is mostly based on experience, and I've set it to 50 as my baseline. The constant can be adjusted, allowing for further tuning. Addtionally, neural networks work best with their targets expressed as integers. We'll accomplish this by _coding_ the categories, using the built-in Pandas method. After coding, each category will have a single corresponding number. For example, 'Active Directory' will be '3'.

Similar to encoding categories to numbers, the input tweets themselves need a representation that is easily understood by the neural networks. This is accomplished by using a _Tokenizer_. Once fit onto a corpus, the _Tokenizer_ will assign a number to each word. This allows us to turn any text into an array of numbers that can be processed by the neural network. Conveniently, Keras included a _Tokenizer_. Our implementation will utilize this Keras function.



### Implementation

In order to perform this classification, I implemented a two-part neural network in Keras. The solution was created and ran inside of a Jupyter Notebook utilizing Tensorflow-GPU. 

The first neural network is a _word2vec_ network that was pre-trained on an extremely large corpus (a set of text). The vectors created from a word2vec network are called _Embeddings_. They are a popular way to represent textual information within machine learning. The created embeddings contain a representation for the word that is learned from the corpus, which can reflect the relationship to other words. In my network, I locked the learned embeddings from being trained. In this case, I'm relying completely on the Google-trained set "GoogleNews-Vectors-negative300". The 300 here is an important value, as it represents the length of each embedding vector. Google provides several embedding vector lengths in pre-trained sets, but any value is possible to use if you train it yourself.

This network has created a mapping of words to embedding vectors, which we then input our entire corpus of text into. In essence, this allows us to represent each word with a 300-length vector. This becomes the first step in our algorithm - take each word from our tweet and represent it as an embedding vector.

Once we've created this vector, we use _Reshape_ to change how we're representing the data. There are various ways to process a sentence after _word2vec_, with the traditional method involving averaging the vector of each word in the sentence. However, we instead transform our embedding vectors into a matrix. Each row of this matrix is one word of the tweet represented as an _embedding vector_. 

This matrix is fed into four separate convolutional towers, each creating filters with different sizes. These filters are each 300 columns with varying number of rows. Each row is representing a word, so you can imagine these filters as covering a different number of words. We used 1, 2, 4, and 5 word filters for each separate tower. Once passed through a filter, the network goes through a _Max Pooling_ to allow for translation of words across the sentence. Additionally, this lowers the computation cost by reducing the number of variables moving forward.

The output of each tower is concatenated together before being fed into another small convolution filter. The output of all of this is put into a _Global Max Pooling_ layer that helps more with translational flexibility and vastly reduces our number of parameters by taking the average of each feature map fed into it.

This network is then fed into a simple fully connected dense layer with 256 nodes. The following layer is a dropout layer to help prevent overfitting to the training dataset. Finally, the network ends with a _softmax_ dense network that outputs the probability for the input to match each category.

### Refinement

The initial pass of the algorithm was surprisingly effective. The first attempt was a sequential network which is much simpler to implement. This was able to predict the category of a tweet with 75% accuracy. However, attempts to increase the accuracy consistently failed. For this reason, the method was switched to a functional multi-column approach. 

In the new architecture, initial training passes have extremely low accuracy. A checkpointer was established with two patiences but the initial couple of training passes never yielded increasing results. The most important breakthrough was to go back to the epoch method and perform a minimum of 10 or more training runs. At two epochs of training, the model yields a lackluster 28% accuracy. However, by epoch 20 we've achieved 92% accuracy. 

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