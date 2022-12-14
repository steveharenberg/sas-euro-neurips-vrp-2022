#include <time.h>
#include <iostream>

#include "Genetic.h"
#include "commandline.h"
#include "LocalSearch.h"
#include "Split.h"
#include "Params.h"
#include "Population.h"
#include "Individual.h"

// Main class of the algorithm. Used to read from the parameters from the command line,
// create the structures and initial population, and run the hybrid genetic search
int main(int argc, char* argv[])
{
	try
	{
		// Reading the arguments of the program
		CommandLine commandline(argc, argv);

		// Reading the data file and initializing some data structures
		// SKIPPING std::cout << "----- READING DATA SET FROM: " << commandline.config.pathInstance << std::endl;
		Params params(commandline);

		// Creating the Split and Local Search structures
		Split split(&params);
		LocalSearch localSearch(&params);

		// Get solutions for warm start
		Solutions const warmstartSols = params.readWarmstartSolutions();

		// Initial population
		// SKIPPING std::cout << "----- INSTANCE LOADED WITH " << params.nbClients << " CLIENTS AND " << params.nbVehicles << " VEHICLES" << std::endl;
		// SKIPPING std::cout << "----- BUILDING INITIAL POPULATION" << std::endl;
		Population population(&params, &split, &localSearch, warmstartSols);

		// Genetic algorithm
		// SKIPPING std::cout << "----- STARTING GENETIC ALGORITHM" << std::endl;
		Genetic solver(&params, &split, &population, &localSearch);
		solver.run(commandline.config.nbIter, commandline.config.timeLimit);
		// SKIPPING std::cout << "----- GENETIC ALGORITHM FINISHED, TIME SPENT: " << params.getTimeElapsedSeconds() << std::endl;

		// Export the best solution, if it exist
		if (population.getBestFound() != nullptr)
		{
			population.getBestFound()->exportCVRPLibFormat(commandline.config.pathSolution);
			population.exportSearchProgress(commandline.config.pathSolution + ".PG.csv", commandline.config.pathInstance, commandline.config.seed);
			if (commandline.config.pathBKS != "")
			{
				population.exportBKS(commandline.config.pathBKS);
			}
		}
		params.timeCost.cleanup();
	}

	// Catch exceptions
	catch (const std::string& e)
	{ 
		// SKIPPING std::cout << "EXCEPTION | " << e << std::endl;
	}
	catch (const std::exception& e)
	{ 
		// SKIPPING std::cout << "EXCEPTION | " << e.what() << std::endl; 
	}

	// Return 0 if the program execution was successful
	return 0;
}
