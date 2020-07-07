# FBP - Facial Beauty Profiling

Whether they've hit or miss over the last 2 years with their fans and critics, the renegade social media giant TikTok has Toosie Slid its way into more than 1 Billion users.

But what determines who makes it onto the ellusive For You Page (#fyp)? Examination of the source code would suggest that a facial beauty metric influences the decision making.

Here we use an AI-based algorithm to make predictions of facial beauty, using a 1-5 rating system.

Data is provided by https://github.com/HCIILAB/SCUT-FBP5500-Database-Release, a 5500-image datase of human faces with associated ratings.

The particularly interesting aspect of this project is that since beauty is of course subjective, our AI has learnt to identify the facial features that the labellers themselves found more and less attractive.

NB: Still a work in progress!

# Instructions for use:

The training code is contained within FBP_classification.py. Training is split into generations to reduce memory requirements.

To run a prediction, place an image of a face in the folder test_cases/, and change the name of the image in call_model.py (the first line in main()) to the desired image (ext included).

Some pre-trained models are included in models/. Enjoy!

