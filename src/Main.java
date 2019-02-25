import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import mpi.MPI;
import jomp.runtime.OMP;

public class Main {

	public static void main(String[] args) {
		final NearestNeighboursClassifier.ExecutionPolicy executionPolicy = NearestNeighboursClassifier.ExecutionPolicy.multiComputerMPJ;
		final int numberOfNeighbours = 11;
		if(executionPolicy == NearestNeighboursClassifier.ExecutionPolicy.multiThreadedJomp || executionPolicy == NearestNeighboursClassifier.ExecutionPolicy.multiThreadedJompMultiComputerMPJ) {
			OMP.setNumThreads(Runtime.getRuntime().availableProcessors());
		}
		if(executionPolicy == NearestNeighboursClassifier.ExecutionPolicy.multiComputerMPJ || executionPolicy == NearestNeighboursClassifier.ExecutionPolicy.multiThreadedJompMultiComputerMPJ){
			if(executionPolicy == NearestNeighboursClassifier.ExecutionPolicy.multiComputerMPJ) {
				MPI.Init(args);
			} else {
				MPI.initThread(MPI.THREAD_MULTIPLE, MPI.THREAD_MULTIPLE, args);
				if(!MPI.Initialized()) {
					System.out.println("Multi threaded mpj not supported");
					return;
				}
			}
			
			int rank = MPI.COMM_WORLD.Rank();
			int size = MPI.COMM_WORLD.Size();
			int[] trainingLength = new int[1];
			Dataset dataset = null;
			if(rank == 0){
				dataset = loadDataset();
				trainingLength[0] = dataset.trainingPredictors.length;
			}
			MPI.COMM_WORLD.Bcast(trainingLength, 0, 1, MPI.INT, 0);
			int[] sendcounts = new int[size];
			int[] displs = new int[size+1];
			displs[0] = 0;
			int remainder = 0;
			int length = trainingLength[0] / 4;
			for(int i = 0; i < size; ++i){
				remainder += length;
				sendcounts[i] = (remainder / size) * 4;
				remainder %= size;
				displs[i+1] = displs[i] + sendcounts[i];
			}
			float[] localTrainingPredictors = new float[sendcounts[rank]];
			int[] localTrainingTypes = new int[sendcounts[rank] / 4];
			if(rank == 0){
				MPI.COMM_WORLD.Scatterv(dataset.trainingPredictors, 0, sendcounts, displs, MPI.FLOAT,
						localTrainingPredictors, 0, sendcounts[rank], MPI.FLOAT, 0);
				for(int i = 0; i < sendcounts.length; ++i){
					sendcounts[i] /= 4;
					displs[i] /= 4;
				}
				
				MPI.COMM_WORLD.Scatterv(dataset.trainingTypes, 0, sendcounts, displs, MPI.INT,
						localTrainingTypes, 0, sendcounts[rank], MPI.INT, 0);
				
				dataset.trainingPredictors = null;
				dataset.trainingTypes = null;
				
				NearestNeighboursClassifier model = new NearestNeighboursClassifier(numberOfNeighbours, executionPolicy)
						.fit(localTrainingPredictors, 4, localTrainingTypes);
				MPI.COMM_WORLD.Bcast(dataset.testPredictors, 0, dataset.testPredictors.length, MPI.FLOAT, 0);
				int[] predictions = model.predict(dataset.testPredictors);
				float score = score(dataset.testTypes, predictions);
				System.out.println(score);
			} else {
				MPI.COMM_WORLD.Scatterv(null, 0, sendcounts, displs, MPI.FLOAT,
						localTrainingPredictors, 0, sendcounts[rank], MPI.FLOAT, 0);
				for(int i = 0; i < sendcounts.length; ++i){
					sendcounts[i] /= 4;
					displs[i] /= 4;
				}
				MPI.COMM_WORLD.Scatterv(null, 0, sendcounts, displs, MPI.INT,
						localTrainingTypes, 0, sendcounts[rank], MPI.INT, 0);
				NearestNeighboursClassifier model = new NearestNeighboursClassifier(numberOfNeighbours, executionPolicy)
						.fit(localTrainingPredictors, 4, localTrainingTypes);
				float[] testPredictors = new float[15 * 4];
				MPI.COMM_WORLD.Bcast(testPredictors, 0, testPredictors.length, MPI.FLOAT, 0);
				model.predict(testPredictors);
			}
			MPI.Finalize();
		} else {
			Dataset dataset = loadDataset();
			NearestNeighboursClassifier model = new NearestNeighboursClassifier(numberOfNeighbours, executionPolicy)
					.fit(dataset.trainingPredictors, 4, dataset.trainingTypes);
			int[] predictions = model.predict(dataset.testPredictors);
			float score = score(dataset.testTypes, predictions);
			System.out.println(score);
		}
	}
	
	private static float score(int[] testTypes, int[] predictions){
		float score = 0.0f;
		for(int i = 0; i < testTypes.length; ++i){
			if(predictions[i] == (testTypes[i])){
				score += 1.0f;
			}
		}
		score /= testTypes.length;
		return score;
	}
	
	private static class Dataset {
		public float[] trainingPredictors;
		public int[] trainingTypes;
		public float[] testPredictors;
		public int[] testTypes; 
		//public List<String> typeDecoder;
		
		public Dataset(float[] trainingPredictors, int[] trainingTypes, float[] testPredictors, int[] testTypes/*,
				List<String> typeDecoder*/){
			
			this.trainingPredictors = trainingPredictors;
			this.trainingTypes = trainingTypes;
			this.testPredictors = testPredictors;
			this.testTypes = testTypes;
			//this.typeDecoder = typeDecoder;
		}
	}
	
	private static Dataset loadDataset() {
		ArrayList<Float> predictors = new ArrayList<>();
		ArrayList<Integer> types = new ArrayList<>();
		List<String> typeDecoder = Arrays.asList(new String[]{"Iris-setosa", "Iris-versicolor", "Iris-virginica"});
		try(BufferedReader br = new BufferedReader(new FileReader("data/iris.data"))) {
		    String line = br.readLine();
		    while (line != null) {
		    	String[] values = line.split(",");
		    	for(int i = 0; i < 4; ++i){
		    		predictors.add(Float.parseFloat(values[i]));
		    	}
		    	types.add(typeDecoder.indexOf(values[4]));
		        line = br.readLine();
		    }
		} catch(Exception e){
			e.printStackTrace();
		}
		
		float[] testPredictors = new float[15 * 4];
		int[] testTypes = new int[15];
		Random rnd = new Random(2456);
		for(int i = 0; i < 5; ++i){
			int index = Math.abs(rnd.nextInt()) % (50 - i) + 100;
			for(int j = 0; j < 4; ++j){
				testPredictors[i * 4 + j] = predictors.get(index * 4 + j);
				predictors.set(index * 4 + j, predictors.get(predictors.size() - 5 + j));
			}
			for(int j = 0; j < 4; ++j){
				predictors.remove(predictors.size()-1);
			}
			testTypes[i] = types.get(index);
			types.set(index, types.get(types.size() - 1));
			types.remove(types.size()-1);
		}
		for(int i = 0; i < 5; ++i){
			int index = Math.abs(rnd.nextInt()) % (50 - i) + 50;
			for(int j = 0; j < 4; ++j){
				testPredictors[(i + 5) * 4 + j] = predictors.get(index * 4 + j);
				predictors.set(index * 4 + j, predictors.get(predictors.size() - 5 + j));
			}
			for(int j = 0; j < 4; ++j){
				predictors.remove(predictors.size()-1);
			}
			testTypes[i+5] = types.get(index);
			types.set(index, types.get(types.size() - 1));
			types.remove(types.size()-1);
		}
		for(int i = 0; i < 5; ++i){
			int index = Math.abs(rnd.nextInt()) % (50 - i);
			for(int j = 0; j < 4; ++j){
				testPredictors[(i + 10) * 4 + j] = predictors.get(index * 4 + j);
				predictors.set(index * 4 + j, predictors.get(predictors.size() - 5 + j));
			}
			for(int j = 0; j < 4; ++j){
				predictors.remove(predictors.size()-1);
			}
			testTypes[i+10] = types.get(index);
			types.set(index, types.get(types.size() - 1));
			types.remove(types.size()-1);
		}
		
		float[] temp = new float[predictors.size()];
		for(int i = 0; i < temp.length; ++i){
			temp[i] = predictors.get(i);
		}
		int[] temp2 = new int[types.size()];
		for(int i = 0; i < types.size(); ++i){
			temp2[i] = types.get(i);
		}
		
		return new Dataset(temp, temp2, testPredictors, testTypes/*, typeDecoder*/);
	}
}
