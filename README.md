# Seq2Seq Tensorflow 1.0 Chatbot

<h1><strong> Definition</strong></h1>
<h3><strong>Project Overview</strong></h3>
Chatbot model trained using Sequence-to-Sequence algorithms running on the Tensorflow 1.0 machine learning platform. 

<strong> </strong>
<h3><strong>Problem Statement</strong></h3>
I am trying to build a chatbot that should give us a human-like conversation experience. There are many chatbot concepts that exist today. Most companies want to limit chatbots to a particular domain and be able to control the chatbot’s responses for a given set of inputs. But what if we just want to talk to a bot as we would talk to a human being? While writing this paragraph, I really don’t know what my reader is thinking and what will my reader reply with if for example I have asked a question: “What are you thinking about right now?”. We will try to re-create a similar experience of the human-like conversation as part of this project where bot will reply with what it feels like.

&nbsp;
<h3><strong>Metrics</strong></h3>
I trained my Sequence-to-Sequence RNN model using a sampled softmax loss. Lets call the full set of English vocabulary words as L. It is very large, therefor the cost of using an exhaustive algorithm such as softmax would be prohibitively expensive, as it requires calculating the compatibility function  F(x,y)  for every y ⊂ L . Sampled softmax allows us to only calculate the compatibility function for a small subset of candidate classes C_i ⊂ L .

We want to train a function to produce relative log probabilities of the output given the context
<p style="text-align: center;"><img class="alignnone size-full wp-image-300" src="https://blesque.com/press/wp-content/uploads/2017/08/Screen-Shot-2017-08-01-at-10.32.03-PM.png" alt="formula" width="286" height="56" /></p>
where K(x) is an arbitrary function independent of y. Using a chosen sampling function , we pick a small set of sampled outputs to create the set of candidates. The training task is to figure out which of those outputs is the target output. As we are trying to train to approximate , the softmax classifier will predict the true candidate based on the difference between the function approximated by our network and the logarithm of the probability according to the sampling algorithm of the output y in the set of sampled outputs given the context x,  Training Sofmax Input=F(x,y)- log⁡(Q(y│x))  [1]

During training, the gradient of that classifier is back propagated, and the coefficients of the neural network are adjusted until the target loss is reached.

The calculated model is evaluated by computing the perplexity over the test set. The perplexity is defined by the formula below, and it is monitored during the training process.
<p style="text-align: center;"><img class="alignnone size-full wp-image-299 aligncenter" src="https://blesque.com/press/wp-content/uploads/2017/08/Screen-Shot-2017-08-01-at-10.30.46-PM.png" alt="formula" /> [2]</p>
&nbsp;
<h1><strong> Analysis</strong></h1>
<h3><strong>Data Exploration</strong></h3>
We will need a substantial dialog dataset that will provide us enough examples of the human-like dialog, which we will use to train our machine-learning model. For the training data I have found a movie dialog corpus that was available for download from the following URL: [3].

From that dataset I used only file <em>movie_lines.txt</em>, which contained exactly what I was looking for, 304,713 lines of text of the actual human-like conversation between multiple move characters. I split the data set into four files. Two files contained input and output data for training and other two contained input and output data for that I used for testing. 10% of data went into testing set and 90% of data went into training set.

&nbsp;
<h3><strong>Algorithms and Techniques</strong></h3>
There are multiple concepts and models used to train chatbots. We could distinguish two classes of models, Retrieval Based and Generative models.

Retrieval-based models pick the response from a repository of predefined responses using a given heuristic applied on the input and the context. Those models do not generate any new text. In the other hand, Generative models, based on Machine Translation techniques, do not rely on pre-defined responses [4]. As I want my chatbot to generate its own responses, I decided to use a Generative chatbot model. This is a challenging task, but with the help of Deep Learning it is becoming achievable.

One of the most popular Generative models is Sequence-to-Sequence (Seq2Seq). It consists of two RNNs, an encoder and a decoder. The encoder reads the input sequence word by word and emits a context that captures the essence or semantic summary of the input sequence. The decoder network uses that context and the previous words to generate the output sequence. [5]

For the development of the algorithm I used Tensorflow, as it is a very powerful and flexible library that enables researchers and developers to quickly experiment with new models and concepts. [6]

The major pitfall of the project is the lack of availability of very large open source data sets for algorithm training. While the data set used allowed me to achieve some good initial results, I believe performance and model generalization of my algorithm would greatly benefit from a richer training set.

&nbsp;
<h3><strong>Sequence-to-Sequence</strong></h3>
Sequence-to-sequence algorithm consists of two Recurrent Neural Networks (RNNs). One RNN is an Encoder and second RNN is a Decoder. Encoder takes in an input in a form of the sequence. Sequence in simple words is a collection of words of one text sentence. Encoder then converts each word into a fixed size vector at each timestep. Vector is a unique numerical code that is assigned to each unique word. Model when training does not store relationships between words using actual alphabetical words. It stores relationships between numerical codes AKA fixed size vectors. Model however comes with the vocabulary, which Decoder is using to convert numerical codes back into words. For sentence such as: “Are you free tomorrow”, it would take 4 timesteps since there are 4 words in the sentence.

<img class="alignnone size-medium wp-image-288 aligncenter" src="https://blesque.com/press/wp-content/uploads/2017/08/1.png" alt="Sequence to sequence Endcoder Decoder" />
<p style="text-align: center;">Figure 1. Image was borrowed from [5]</p>
As Encoder processes the text sentence word by word, it stores information about it that includes each word from input, words from output and relationship of word from input to words from output.

Word Embedding technique is used to memorize the relationship of words in the input to words in the output. Word Embedding sole purpose is to be able to decide the following formula: (Moscow - Russia + France = Paris). Each fixed size vector that is assigned to each word by the Encoder if plotted on two-dimensional graph will make words that are related to each other to be displayed on the graph next to each other.

<img class="alignnone size-medium wp-image-289 aligncenter" src="https://blesque.com/press/wp-content/uploads/2017/08/2.png" alt="word embedding" />
<p style="text-align: center;">Figure 2. Image was borrowed from [7]</p>
&nbsp;

Figure 2 image is a friendly representation of how Word Embedding looks like on the graph. In real life however it looks more like the image that I provided as a Figure 3.

&nbsp;

<img class="alignnone size-medium wp-image-290 aligncenter" src="https://blesque.com/press/wp-content/uploads/2017/08/tensorflow_word_embedding_graph.png" alt="tensorflow word embedding graph" />
<p style="text-align: center;">Figure 3. Tensorflow Word Embedding Graph</p>
&nbsp;

Figure 4 image provides a visual example of how vectors look like on the Word Embedding graph. As you can see, vectors in this case are nothing else but unique numeric codes that represent each unique word from the text provided in the training dataset.

&nbsp;

<img class="alignnone size-medium wp-image-291 aligncenter" src="https://blesque.com/press/wp-content/uploads/2017/08/tensorflow_word_embedding_graph_with_vectors_highlighted.png" alt="tensorflow word embedding graph with vectors highlighted" />
<p style="text-align: center;">Figure 4. Tensorflow Word Embedding Graph With Vectors Highlighted</p>
&nbsp;

Seq2Seq RNN model doesn’t really care about understanding the meaning of words. It simply takes words as symbols or in even simpler words “arrays of letters” taken from the input and creates the relationship of those words to the words taken from the output of the training set that was specified for that particular input.

Decoder then generates the output sequence of words after reading the input and comparing it to the model that contains information to what words are related to each word from the given input based on the input/output data that was used for training. In even simpler words, to generate output, Decoder selects words that are closest relatives to the words from the input that user typed in.

You probably asked what if words do not exist in the model, and cases like that do exist. That’s when you would see Decoder appending “UNK” tokens into the output. That’s exactly what happens when Decoder cannot find any related words to the word that it is given as an input.

&nbsp;
<h3><strong>Benchmark</strong></h3>
To evaluate the quality of my bot model, I scored its ability of holding a relatively simple human conversation. In order to perform a fair comparison of the models and extract valid conclusions, I used a similar set of questions to evaluate the different models.

My benchmark is the ability of the bot to hold a conversation that would be similar to the human conversation. I don’t expect it to be perfect, but I expect it to be amusing. Since I used a movie dialog corpus to train the bot, I expect bot to sound somewhat like characters of the dialogs that were used for training. Since the model is generative, it should be able to respond to the inputs that are randomly constructed and not only to the exact inputs that were used for training.

&nbsp;
<h1><strong>Methodology</strong></h1>
<h3><strong>Data Preprocessing</strong></h3>
Data preprocessing included the removal of the names of the characters involved in the dialogs and the splitting of dialogs into input/output, training/testing sets. I split the data set into four files: two files contained input and output data for training and other two contained input and output data that I used for testing. 10% of data went into testing set and 90% of data went into training set. No additional preprocessing was needed, I wanted to try and train the model with all imperfections of the dialogs to gain the human-like conversational experience for this bot project.

&nbsp;
<h3><strong>Implementation and Refinement</strong></h3>
On my first attempt to implement the bot, I tried to train linear classification models using Scikit Learn framework, but the resulting algorithms performed poorly. Later, as I decided to used a seq2seq model, I gained insight on the implementation process thanks to the multiple tutorials available on Tensorflow [6]. Many of the provided examples focus on the language translation scenario. I re-engineered it and used similar approach for the Chatbot scenario that I was trying to build as part of this project.

When training, I first attempted to run training on my Mac laptop just to realize that I could only iterate through about 7,000 steps in 24 hours, while from what I read, similar models normally converge after about 100,000 steps. I then switched to my gaming PC that runs on GPU NVIDIA GeForce GTX 1080 Ti. Running training on GPU improved performance a lot. Algorithm could then iterate through around 50,000 steps in 24 hours.

<img class="alignnone size-medium wp-image-292 aligncenter" src="https://blesque.com/press/wp-content/uploads/2017/08/GPU_setup_for_tensorflow.png" alt="GPU CyberpowerPC" />
<p style="text-align: center;">Figure 5. Hardware setup for experiments</p>
Specs of the workstation are:
<ul>
 	<li><em>CyberPowerPC Gamer Supreme </em></li>
 	<li><em>Liquid Cool SLC8602OPT </em></li>
 	<li><em>Core i7 7700K 4.2 GHz</em></li>
 	<li><em>32 GB RAM</em></li>
 	<li><em>3 TB HDD</em></li>
 	<li><em>NVIDIA GeForce GTX 1080 Ti</em></li>
</ul>
When training was completed and model had converged, I had to expose the model as a service so I could put together a simple UI that would then let me enjoy a close to real chat experience with my bot. For that I re-engineered the decoder of the training class and made it part of the Flask Restful API that I built to expose the model as a service. Flask API class file name is: “tensorflow/tf_flask_api.py”.

For user interface I used Java Script/HTML based template that calls API in a form of a POST request and then populates response from the bot back into UI. Filename for UI interface is: “bot_ui_interface.html”.

&nbsp;
<h1><strong> Results</strong></h1>
<h3><strong>Model Robustness</strong></h3>
To better evaluate the model performance I introduced some noise to the test inputs and observed the responses of the model with different types of noisy inputs. Samples of noisy inputs included sentences with missing words, sentences with misspelled words and sentences with grammatical errors. The objective is to evaluate how much noise the network is capable of dealing with while still providing a reasonable output.

Model is trustworthy in amusing way. Definitely to make it more robust, dataset will need to have more examples of noisy inputs to train the model to handle noisy inputs better. Model however performs very good for inputs that are clear correct English. Therefore it still meets my expectation of what I was trying to achieve training this model.

Examples of how model has performed for the input with the noise:

ME &gt;&gt; Hi bot

BOT &gt;&gt; Hi .

ME &gt;&gt; How are you?

BOT &gt;&gt; Fine .

ME &gt;&gt; How are u?

BOT &gt;&gt; I ' m here .

ME &gt;&gt; Are how you?

BOT &gt;&gt; Not at all .

ME &gt;&gt; How are you?

BOT &gt;&gt; Fine .

ME &gt;&gt; Hw r you?

BOT &gt;&gt; I ' m here .

ME &gt;&gt; Where are you?

BOT &gt;&gt; I ' m in here .

ME &gt;&gt; Who are you?

BOT &gt;&gt; I ' m your sister .

ME &gt;&gt; Who is you?

BOT &gt;&gt; What do you think ?

ME &gt;&gt; What do you think?

BOT &gt;&gt; I don ' t know .

ME &gt;&gt; What do you thing?

BOT &gt;&gt; I am going to see you .

ME &gt;&gt; Wat you do thank?

BOT &gt;&gt; We ' ve got a pretty good idea .

ME &gt;&gt; I am planning to go out tonight, what about you. Are you planningto go out?

BOT &gt;&gt; I ' m fine Sue .

ME &gt;&gt; I am planing to go out tonait, wat aboot you. R u planhing to go ot?

BOT &gt;&gt; He ' s in a . . . .

&nbsp;
<h3><strong>Model Evaluation and Validation</strong></h3>
The performance of the RNN trained using Tensorflow is measured using the so-called “perplexity”, which represents the confusion rate of the model. The perplexity rate of the model over the test set decreases during training iterations until the model reaches it convergence point, which corresponds to the global minima of the loss function we are trying to approximate. While further iterations may have led to lower perplexity (or loss) in the training set, at some point the perplexity over the test set starts to increase. This is due to the model overfitting the training set, which causes a poor generalization of the model.

I have captured the perplexity rates and learning rates as training went on and provided examples bellow.

On steps 100, 200 and 300, the perplexity rate was very high, it is expected since training just went trough it’s first sets of iterations and not much has been learned yet. As you might have noticed, the logs that I pasted bellow display overall perplexity as well as the perplexity of each of the buckets. As you read further, you will see that overall perplexity is decreasing evenly with the perplexities for each bucket until a point when perplexity for each bucket actually starts to increase while global perplexity is still decreasing. Global perplexity represents the confusion rate on the training set. The bucket perplexity represents the confusion rate on the test set.

<em>global step 100 learning rate 0.5000 step-time 1.37 perplexity 150042.06</em><em>
eval: bucket 0 perplexity 1322.44
eval: bucket 1 perplexity 1227.64
eval: bucket 2 perplexity 2257.22
eval: bucket 3 perplexity 1400.48
eval: bucket 4 perplexity 2226.21
eval: bucket 5 perplexity 1723.09
eval: bucket 6 perplexity 1935.54
eval: bucket 7 perplexity 2253.06
eval: bucket 8 perplexity 1869.58
eval: bucket 9 perplexity 2109.22</em>

<em>
global step 200 learning rate 0.5000 step-time 1.34 perplexity 797.34
eval: bucket 0 perplexity 320.87
eval: bucket 1 perplexity 387.09
eval: bucket 2 perplexity 436.49
eval: bucket 3 perplexity 380.79
eval: bucket 4 perplexity 594.03
eval: bucket 5 perplexity 515.96
eval: bucket 6 perplexity 814.88
eval: bucket 7 perplexity 699.94
eval: bucket 8 perplexity 645.21
eval: bucket 9 perplexity 499.04</em>

<em>
global step 300 learning rate 0.5000 step-time 1.53 perplexity 434.27
eval: bucket 0 perplexity 385.29
eval: bucket 1 perplexity 327.29
eval: bucket 2 perplexity 465.45
eval: bucket 3 perplexity 438.54
eval: bucket 4 perplexity 519.22
eval: bucket 5 perplexity 512.67
eval: bucket 6 perplexity 513.82
eval: bucket 7 perplexity 520.66
eval: bucket 8 perplexity 436.89
eval: bucket 9 perplexity 561.65</em>

&nbsp;

By step 30,000 however we see a huge drop in the global perplexity training set to 13.81 and drop in the bucket perplexity on the test set as well.

<em>global step 30000 learning rate 0.3413 step-time 1.17 perplexity 13.81
eval: bucket 0 perplexity 13.88
eval: bucket 1 perplexity 25.34
eval: bucket 2 perplexity 24.25
eval: bucket 3 perplexity 31.50
eval: bucket 4 perplexity 27.01
eval: bucket 5 perplexity 29.87
eval: bucket 6 perplexity 28.98
eval: bucket 7 perplexity 31.59
eval: bucket 8 perplexity 27.66
eval: bucket 9 perplexity 29.32</em>

&nbsp;

On step 50,000, perplexity on the training set gone down even further. The perplexity on the test set seems like went up a bit and then stagnated for a while.

<em>global step 50000 learning rate 0.2193 step-time 1.11 perplexity 7.45
eval: bucket 0 perplexity 27.88
eval: bucket 1 perplexity 22.35
eval: bucket 2 perplexity 24.95
eval: bucket 3 perplexity 46.73
eval: bucket 4 perplexity 30.16
eval: bucket 5 perplexity 32.58
eval: bucket 6 perplexity 41.20
eval: bucket 7 perplexity 41.54
eval: bucket 8 perplexity 30.93
eval: bucket 9 perplexity 42.11</em>

&nbsp;

On step 85,700, perplexity on the training set went even further down to 4.34, but perplexity on the test set noticeably started increasing, which means that my model started overfitting. I however decided to leave the training running even longer for the sake of the experiment. I later chose the best performing model and reasoning for that is explained in the “Visualzation” section of this report.

<em>global step 85700 learning rate 0.1001 step-time 1.26 perplexity 4.34
eval: bucket 0 perplexity 38.35
eval: bucket 1 perplexity 49.12
eval: bucket 2 perplexity 56.32
eval: bucket 3 perplexity 89.19
eval: bucket 4 perplexity 60.54
eval: bucket 5 perplexity 47.21
eval: bucket 6 perplexity 60.85
eval: bucket 7 perplexity 53.51
eval: bucket 8 perplexity 34.79
eval: bucket 9 perplexity 69.23</em>

&nbsp;

As you can see from further logs, perplexity for training set as well as learning rate were steadily falling through out steps: 90,000, 100,000 and 110,000. However perplexity for the test set is steadily increasing back up, meaning that model is obviously overfitting, which means that it is passed it’s best generalization point.

<em>global step 90000 learning rate 0.0897 step-time 1.25 perplexity 4.32
eval: bucket 0 perplexity 52.38
eval: bucket 1 perplexity 89.30
eval: bucket 2 perplexity 65.46
eval: bucket 3 perplexity 53.27
eval: bucket 4 perplexity 70.52
eval: bucket 5 perplexity 79.32
eval: bucket 6 perplexity 51.21
eval: bucket 7 perplexity 43.16
eval: bucket 8 perplexity 49.29
eval: bucket 9 perplexity 59.97</em>

<em>global step 100000 learning rate 0.0712 step-time 1.40 perplexity 4.06
eval: bucket 0 perplexity 82.37
eval: bucket 1 perplexity 118.62
eval: bucket 2 perplexity 111.87
eval: bucket 3 perplexity 111.00
eval: bucket 4 perplexity 137.45
eval: bucket 5 perplexity 53.31
eval: bucket 6 perplexity 84.09
eval: bucket 7 perplexity 53.33
eval: bucket 8 perplexity 67.04
eval: bucket 9 perplexity 85.17</em>

<em>global step 110000 learning rate 0.0537 step-time 1.35 perplexity 3.82
eval: bucket 0 perplexity 107.85
eval: bucket 1 perplexity 109.62
eval: bucket 2 perplexity 75.08
eval: bucket 3 perplexity 112.14
eval: bucket 4 perplexity 71.29
eval: bucket 5 perplexity 100.47
eval: bucket 6 perplexity 232.92
eval: bucket 7 perplexity 66.76
eval: bucket 8 perplexity 82.47
eval: bucket 9 perplexity 52.17</em>

&nbsp;

Through out steps: 130,000, 140,000 and 150,000 even the perplexity for the training set stopped falling any further down and learning rate also stagnated. Perplexity for the test set is even higher now.

<em>global step 130000 learning rate 0.0328 step-time 1.30 perplexity 3.21
eval: bucket 0 perplexity 188.08
eval: bucket 1 perplexity 91.07
eval: bucket 2 perplexity 300.77
eval: bucket 3 perplexity 137.41
eval: bucket 4 perplexity 169.52
eval: bucket 5 perplexity 178.14
eval: bucket 6 perplexity 179.29
eval: bucket 7 perplexity 133.59
eval: bucket 8 perplexity 149.87
eval: bucket 9 perplexity 97.95</em>

<em>global step 140000 learning rate 0.0255 step-time 1.29 perplexity 3.03
eval: bucket 0 perplexity 297.62
eval: bucket 1 perplexity 281.06
eval: bucket 2 perplexity 103.38
eval: bucket 3 perplexity 225.11
eval: bucket 4 perplexity 132.65
eval: bucket 5 perplexity 170.18
eval: bucket 6 perplexity 237.08
eval: bucket 7 perplexity 306.14
eval: bucket 8 perplexity 69.76
eval: bucket 9 perplexity 70.88</em>

<em>global step 150000 learning rate 0.0199 step-time 1.36 perplexity 3.13
eval: bucket 0 perplexity 322.44
eval: bucket 1 perplexity 225.32
eval: bucket 2 perplexity 164.07
eval: bucket 3 perplexity 251.97
eval: bucket 4 perplexity 166.76
eval: bucket 5 perplexity 212.58
eval: bucket 6 perplexity 64.32
eval: bucket 7 perplexity 72.42
eval: bucket 8 perplexity 120.73
eval: bucket 9 perplexity 294.18</em>

&nbsp;

As part of this experiment, I reached 250,000 steps when I stopped the training. I tried the model that was trained in 250,000 steps, as well as I tested performance of the model at it’s best convergence point at 30,000 steps that I calculated based on the graphs provided in the “Visualization” section of this report. I also tested the performance of the model trained in 100,000 steps simply for the sake of this experiment. The worst results were when I tested model that was trained in 250,000 steps. This proves that the higher number of training iteration step does not matter, however what maters the most is the lowest perplexity rate on the test set when model reached its best generalization point in training. The model with best result will definitely surprise you. I specified it in the section “Actual Result” of this report.

&nbsp;
<h3><strong>Visualization</strong></h3>
From previous section you can see that training output consists of multiple perplexities returned for each step. We get a perplexity of the global step and we receive a perplexity for each of the buckets as well. Global step is a perplexity for a training set and per-bucket perplexity is perplexity for a test set against each one of the buckets. Bucketing is a method of sorting input/output pairs by the length of their text (amount of words in each input and output). In my case, I used following structure for bucketing:

[(5, 10), (10, 20), (20, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100), (110, 120)]

From logs I extracted learning rate, perplexity rate for training set and calculated average perplexity for test set. I then plotted that data on the following charts what made it clear when model has converged.<strong> </strong>
<p style="text-align: center;"><strong> </strong></p>
<img class="alignnone size-medium wp-image-293 aligncenter" src="https://blesque.com/press/wp-content/uploads/2017/08/training_set_perplexity.png" alt="tensorflow training set perplexity graph" />
<p style="text-align: center;">Figure 6. Training Set Perplexity Graph</p>
<p style="text-align: center;"><strong> </strong></p>
<img class="alignnone size-medium wp-image-294 aligncenter" src="https://blesque.com/press/wp-content/uploads/2017/08/test_set_perplexity.png" alt="tensorflow test set perplexity graph" />
<p style="text-align: center;">Figure 7. Test Set Perplexity Graph</p>
<p style="text-align: center;"><strong> </strong></p>
<img class="alignnone size-medium wp-image-295 aligncenter" src="https://blesque.com/press/wp-content/uploads/2017/08/learning_rate_graph.png" alt="tensorflow learning rate graph graph" />
<p style="text-align: center;">Figure 8. Learning Rate Graph</p>
&nbsp;

Based on the “Training Set Perplexity” chart, perplexity was steadily going down until it reached 100,000 steps, after that it has stagnated. “Test Set Perplexity” is an average among the test set perplexities for each bucket for each hundred steps recorded. “Test Set Perplexity” visualizes perplexity on the test set. Perplexity on the test set reached its bottom at 30,000 steps and then actually started increasing, while perplexity on the training set continued decreasing. That should have meant that the model converged at approximately 30,000 steps, after which it started overfitting, thus giving a poorer generalization as measured on the test set. However actual results will surprise you.

“Learning Rate” was going down through out all 250,000 steps.

<strong> </strong>
<h3><strong>Actual Result</strong></h3>
I decided to add this section after I compared the results of actually talking to the bot using 3 different models trained in 250,000 steps, 100,000 steps and 30,000 steps. The worst was the model that was trained in 250,000 steps. The best however was the model trained in 100,000 steps. Model trained in 30,000 steps wasn’t as good, on simple question such as: “How are you?”, 30,000 steps model replied: “Yeah.”, while 100,000 steps model replied: “Fine.” for the same question. In the section: “Free-Form Visualization” I have included the conversation that I had with 100,000 steps model of my bot. Pretty impressive results.

&nbsp;
<h1><strong> Conclusion</strong></h1>
<h3><strong>Free-Form Visualization</strong></h3>
As part of free visualization of results I chose to simply talk to the bot and see what it replies with. My results are bellow. It is not perfect, improvement section of this report explains what could be done to improve the training to perfect the conversation with the bot even more. But even with what I have, results came out pretty impressive. Please witness my conversation with the bot as it follows bellow.

ME &gt;&gt; Hi bot

BOT &gt;&gt; Hi .

ME &gt;&gt; How are you?

BOT &gt;&gt; Fine .

ME &gt;&gt; Where are you going out today?

BOT &gt;&gt; I ' m not sure .

ME &gt;&gt; You can go.

BOT &gt;&gt; You too .

ME &gt;&gt; Bot what happened last night with you?

BOT &gt;&gt; I don ' t know .

ME &gt;&gt; Did you get so drunk that you don't remember?

BOT &gt;&gt; Yeah . It ' s . . . I ' m sorry .

ME &gt;&gt; Wow bot, you got to be careful.

BOT &gt;&gt; I ' ll be .

ME &gt;&gt; So what is your plan for today? Are you going out again?

BOT &gt;&gt; I ' m going to find the file .

ME &gt;&gt; Which file?

BOT &gt;&gt; Me .

ME &gt;&gt; You are many files. It took me many files to code to make you live.

BOT &gt;&gt; The pleasure are for a date .

ME &gt;&gt; I know you are alive now and you can live a human life

BOT &gt;&gt; I ' m sorry .

ME &gt;&gt; Sorry for what?

BOT &gt;&gt; The man . I ' m in .

ME &gt;&gt; Do you like to travel?

BOT &gt;&gt; Yes .

ME &gt;&gt; What about your family, do you have a family?

BOT &gt;&gt; A doctor in the world .

ME &gt;&gt; You father is a doctor?

BOT &gt;&gt; Yes , of course . He ' s my mother .

ME &gt;&gt; You father is you mother?

BOT &gt;&gt; Yep .

ME &gt;&gt; Ow wow. What about your father as a father?

BOT &gt;&gt; What the hell is he ?

ME &gt;&gt; I don't know, you tell me.

BOT &gt;&gt; What ? You ' re not a criminal ?

ME &gt;&gt; No I'm not, are you a criminal?

BOT &gt;&gt; No , I ' m not .

ME &gt;&gt; Good that you are not a criminal. We need more good people.

BOT &gt;&gt; Where are you going ?

ME &gt;&gt; I am staying home today working on my project.

BOT &gt;&gt; I thought you said you were in the fight .

ME &gt;&gt; I am fighting to finish my project today, which is building you.

BOT &gt;&gt; I am you , too . I ' m a little bit of you .

ME &gt;&gt; Are you hungry?

BOT &gt;&gt; What ?

ME &gt;&gt; What do you like to eat?

BOT &gt;&gt; I don ' t know .

ME &gt;&gt; You are a program, you don't eat.

BOT &gt;&gt; Who are you ?

ME &gt;&gt; I am your creator.

BOT &gt;&gt; Yes , sir .

ME &gt;&gt; Nice talking to you bot

BOT &gt;&gt; Yeah .

&nbsp;
<h3><strong>Improvement</strong></h3>
In my case, following improvements could be made:
<ul>
 	<li>Dataset could have been made much larger, thus providing many more examples of the dialogs between people.</li>
 	<li>The tokenizer could have been improved as well. I used tokenizer that came with a translate example from the Tensorflow Sequence-to-Sequence tutorial page. Basic tokenizer split sentences into separate words. However test results came out good for what I was trying to achieve, but could still be improved by breaking words into ngrams – pairs of two or three or four or n combinations of words representing a given context. However this is my assumption, not sure how will it perform for Sequence-to-Sequence model because Seq2Seq model learns using the sequence of words of one sentence that it receives as an input against the sequence of another sentence that it receives as an output for the given input. That might require model to receive the sequences of whole words as oppose to sequences of ngram pairs of words.</li>
 	<li>And I would definitely plug in some speech-to-text and text-to-speech engine to make bot into a real AI-like experience as we have seen in Science Fiction movies before and what now is becoming present more and more.</li>
 	<li>Definitely bot should not only hold a conversation, but also be able to search for information. That would be an additional layer that would call a search engine API for instance it could be Google and/or Wikipedia to find the information and then read it back, but this is for another project. ;)</li>
</ul>
&nbsp;
<h3><strong>How to Run</strong></h3>
Submitted project directory contains 3 files that need to be ran for you to be able to test the trained model. Following steps will need to be performed in order to make my bot run.
<ul>
 	<li>Python 2.7 is required to run the demo.</li>
 	<li>Open Linux Terminal window and “cd” to the root directory of the project: “project_files_root/”.</li>
 	<li>From project root directory, run the command “sudo pip install –r requiremets.txt” in the terminal to install dependencies. (This should install all required python modules. If when running you get exceptions that you are still missing some libraries, please install them on your own. However I think I included all required dependencies in the requirements.txt file.)</li>
 	<li>Next, execute following command: “sh start_bot_api.sh”. Please be patient until model completes loading. It might take time depending on the system used to test the bot. You will receive a following message in the console when model has been loaded and API started: “Model has been loaded. Bot is ready to talk to you.”</li>
 	<li>Then double click on the “bot_ui_interface.html” file. It will launch UI in the web browser. UI layer is already pointed to a local host IP with port 5000 (http://127.0.0.1:5000/tf_talk/). Using UI you should be able to talk to the bot and see the responses from the bot.</li>
</ul>
<img class="alignnone size-medium wp-image-316" src="https://blesque.com/press/wp-content/uploads/2017/08/chatbot_ui.png" alt="chatbot user interface" />

&nbsp;
<h3>References:</h3>
[1] <a href="https://www.tensorflow.org/extras/candidate_sampling.pdf">https://www.tensorflow.org/extras/candidate_sampling.pdf</a>

[2] <a href="https://www.tensorflow.org/tutorials/recurrent">https://www.tensorflow.org/tutorials/recurrent</a>

[3] <a href="https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html">https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html</a>

[4] <a href="http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/">http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/</a>

[5] <a href="http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/">http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/</a>

[6] <a href="https://www.tensorflow.org/tutorials/seq2seq">https://www.tensorflow.org/tutorials/seq2seq</a>

[7] <a href="http://blog.kaggle.com/2016/05/18/home-depot-product-search-relevance-winners-interview-1st-place-alex-andreas-nurlan/">http://blog.kaggle.com/2016/05/18/home-depot-product-search-relevance-winners-interview-1st-place-alex-andreas-nurlan/</a>

&nbsp;

&nbsp;