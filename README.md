This file will explain how to reproduce the results I was able to generate using my inference time agent.
To generate the output JSON, on the test_data questions:
1) Open the file called "generate_answer_template.py"
2) Open a terminal window in the same working directory
3) run "python generate_answer_template.py"

What I have done, is put all of my agent related code into the generate_answer_template.py file,
and I wrote the agent_loop method to basically call the necessary sub-methods to process the question as needed.

You can also test certain parts of the agent if necessary.
To do so:
1) Open the file called agent.ipynb
2) Open the file named random.ipynb
    This file contains different scripts to test the router layer, domain specific questions, and accuracy checking on dev_data
3) Copy the script for the test you want to run and paste in a new cell in agent.ipynb
4) Run all cells in order and the script cell last
5) The output will begin the print based on the test that was run