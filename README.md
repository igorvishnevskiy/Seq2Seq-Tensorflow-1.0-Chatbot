# Seq2Seq Tensorflow 1.0 Chatbot

<h3><strong>Project Description</strong></h3>
Chatbot model trained using Sequence-to-Sequence algorithms running on the Tensorflow 1.0 machine learning platform. 

<h3><strong>Complete Documentation</strong></h3>
Complete documentation could be found on my blog post that I have created specifically for this project.
Click here for my blog article: <a href="https://blesque.com/press/building-sequence-to-sequence-chatbot-with-tensorflow-1-0/">Building Sequence-to-Sequence Chatbot With Tensorflow 1.0</a>.

<h3><strong>How to Train</strong></h3>
Submitted project directory contains 3 files that need to be ran for you to be able to test the trained model. Following steps will need to be performed in order to make my bot run.
<ul>
 	<li>Python 2.7 is required to run the demo.</li>
 	<li>Open Linux Terminal window and “cd” to the root directory of the project: “project_files_root/”.</li>
 	<li>From project root directory, run the command “sudo pip install –r requiremets.txt” in the terminal to install dependencies. (This should install all required python modules. If when running you get exceptions that you are still missing some libraries, please install them on your own. However I think I included all required dependencies in the requirements.txt file.)</li>
 	<li>Next, execute following command: “sh train_bot.sh”. Please be patient until all variables are loaded and training has started. You will see perplexity output in the console for every 100 steps of training.”</li>
 	<li>I set the training limit to 100000 steps. However steps limit value and the rest of the values for the size of the model, learning_rate, num_layers, and more can be edited in the following file: "project_files_root/tensorflow/flags.py" </li>
 	<li>Training should start right away using the training data that I already have provided in the directory: "project_files_root/tensorflow/combined_data/".</li>
 	<li>When re-training, make sure to empty following directories: "project_files_root/tensorflow/logs/", "project_files_root/tensorflow/model_out/" and "project_files_root/tensorflow/temp_data/". Otherwise training will start where it left off. This is good to have in case you would like to stop training, then continue training further from the point where you stopped it.</li>
	<li>When training will start, missing directories will be added automatically.</li>
</ul>

<h3><strong>How to Run</strong></h3>
Submitted project directory contains 3 files that need to be ran for you to be able to test the trained model. Following steps will need to be performed in order to make my bot run.
<ul>
 	<li>Python 2.7 is required to run the demo.</li>
 	<li>Open Linux Terminal window and “cd” to the root directory of the project: “project_files_root/”.</li>
 	<li>From project root directory, run the command “sudo pip install –r requiremets.txt” in the terminal to install dependencies. (This should install all required python modules. If when running you get exceptions that you are still missing some libraries, please install them on your own. However I think I included all required dependencies in the requirements.txt file.)</li>
 	<li>Next, execute following command: “sh start_bot_api.sh”. Please be patient until model completes loading. It might take time depending on the system used to test the bot. You will receive a following message in the console when model has been loaded and API started: “Model has been loaded. Bot is ready to talk to you.”</li>
 	<li>Then double click on the “bot_ui_interface.html” file. It will launch UI in the web browser. UI layer is already pointed to a local host IP with port 5000 (http://127.0.0.1:5000/tf_talk/). Using UI you should be able to talk to the bot and see the responses from the bot.</li>
</ul>