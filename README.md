# neural cellular automata

### biological inspiration
cells are governed by an internal set of instructions (dna) which determines what state they are in, and this state is influenced by the chemical gradients of the cells around them.

with this project, i intend to mimic this; using an initial "perception" layer which approximates the gradients of the cells, and learning the best update kernel to most closely mimic the training data (a 32x32 pixel image).

### personal motivations
i would really like to learn more about how robust neural nets are built; gaining intuition for the number of layers, the types of layers, and important pre-processing of training data. 

this is inspired by the work done here: [[https://distill.pub/2020/growing-ca/|Growing Neural Cellular Automata]]
