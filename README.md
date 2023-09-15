# PyTorch
In this project I have explored and identified the Design and Security Patterns followed by the JIT compiler module in PyTorch Library.

I have also added a new feature where we can visualize the difference between PyTorch and TorchScript runtimes.

The proposed feature will allow the user to find out the difference between pyTorch runtime and TorchScript runtimes.

I have developed a torch_script_profiler. This profiler takes native models and torch script models as input. Profiler provides an API call `run` which can be used to run both models on a given set of arguments. Each run will consist of an “n” number of executions. We compute the mean runtime of each pass for both models. This Mean runtime is saved in an array which can later be used to generate a plot for users to visualize the difference between runtimes. “run” API takes a number of executions and requirements as input to generate a plot(done using matplotlib), so that these parameters can be easily configured by end users.

To utilize this functionality of torch.jit, i have picked a sample model called BERT.BERT(Bidirectional Encoder Representation) Model is a deep learning model in which every output
element is connected to every input element, and the weightings between them are dynamically calculated based upon their connections.

Step 1: place this TorchScriptProfiler class under the jit module
Ex: pytorch/pytorch/torch/jit/mobile/torch_script_profiler.py
Step 2: place the test class bert_test.py under jit module Ex: pytorch/pytorch/torch/jit/mobile/bert_test.py

The results of the newly added component will show the graphical representation of the difference to train a model in pytorch runtime and torschscript runtime in the following way:
 8
These are the print statements that we included in our code to check the working of TorchScriptProfiler component. These print statements are for training a model for 10 iterations. In 1st iteration each model will be trained once, in second each model will be trained twice and the mean time taken to train the models will be collected. This process repeats until each model gets trained 10 times .
The above 4 implementations were implemented as part of G4 assignment.
   
Details about JIT compiler: https://pytorch.org/docs/stable/jit.html

