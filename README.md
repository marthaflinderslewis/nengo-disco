# nengo-disco
This code gives a toy demonstration of compositional distributional semantics in Nengo. 
This shows adjective-noun composition applied to the pet fish problem. 
The code implement an adjective 'pet', and applies it to successive nouns. 
The most similar noun to the the pet-noun composition is then retrieved. 
We see that a pet fish is rendered as a goldfish, a pet cat as a cat, and a pet lion as a dog.
Note that this is a purely toy model for demonstration purposes only.

## Usage:
You will need to install `nengo`, `nengo-spa`, and `nengo-gui`. After installing these,run `nengo` in the terminal to open the nengo GUI, and then run `pet_fish_model.py` in the GUI. 


