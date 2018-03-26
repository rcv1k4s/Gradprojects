
************************PyCache Simulator with Farthest in Future replacement policy***************************8

Cache Block farthest in known trace is replaced in way number

Run : ./Cachesimulator trace.gz trace.gz 16 32 4  #16 16Kb Cache Size; 32B Block Size; 4 way Associativity

LRU had 25% error rate for first 50k accesses, But FIF Farthest in Future has 15% error.

Can be run on Huge addresses with Psuedo Farthest in Future i.e to replace block according to LRU if No block is found in known future trace,
But needs a little modification to accept file as .gz extension and read N address ahead in to buffer


Ramachandra Vikas Chamarthi
Graduate Research Assistant
The UNC Charlotte
vikaschamarthi240@gmail.com
