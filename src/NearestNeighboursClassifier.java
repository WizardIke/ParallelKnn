import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map.Entry;

import mpi.Datatype;
import mpi.MPI;
import mpi.Op;

import java.util.PriorityQueue;

import jomp.runtime.OMP;

import java.util.Comparator;

public class NearestNeighboursClassifier {
	
	public enum ExecutionPolicy {
		singleThreaded,
		multiThreadedJomp,
		multiComputerMPJ,
		multiThreadedJompMultiComputerMPJ,
	}
	
	private static class CategoryWithDistance {
		public int category;
		public float distance;
		
		public CategoryWithDistance(int category, float distance){
			this.category = category;
			this.distance = distance;
		}
	}
	
	private static class CategoryWithDistanceCompare implements Comparator<CategoryWithDistance>{
		@Override
		public int compare(CategoryWithDistance lhs, CategoryWithDistance rhs){
			if(rhs.distance - lhs.distance < 0.0f){
				return -1;
			} else if(rhs.distance - lhs.distance == 0.0f){
				return 0;
			} else {
				return 1;
			}
		}
	}
	
	private static float euclidianDistance(float[] p1, int p1Offset, float[] p2, int p2Offset, int length){
		float sum = 0.0f;
		for(int i = 0; i < length; ++i){
			float distance = p1[i + p1Offset] - p2[i + p2Offset];
			sum += distance * distance;
		}
		return (float)Math.sqrt(sum);
	}
	
	private static PriorityQueue<CategoryWithDistance> kNearestByDistance(float[] testPredictor, int testPredictorOffset, float[] predictors, int predictorsStride,
			int[] outputs, int k){
		
		PriorityQueue<CategoryWithDistance> best = new PriorityQueue<>(k, new CategoryWithDistanceCompare());
		
		if(outputs.length > k){
			for(int i = 0; i < k; ++i){
				float distance = euclidianDistance(testPredictor, testPredictorOffset, predictors, i * predictorsStride, predictorsStride);
				best.add(new CategoryWithDistance(outputs[i], distance));
			}
			
			for(int i = k; i < outputs.length; ++i){
				float distance = euclidianDistance(testPredictor, testPredictorOffset, predictors, i * predictorsStride, predictorsStride);
				if(distance < best.peek().distance){
					best.poll();
					best.add(new CategoryWithDistance(outputs[i], distance));
				}
			}
			
			return best;
		} else {
			for(int i = 0; i < outputs.length; ++i){
				float distance = euclidianDistance(testPredictor, testPredictorOffset, predictors, i * predictorsStride, predictorsStride);
				best.add(new CategoryWithDistance(outputs[i], distance));
			}
			return best;
		}
	}
	
	private static int[] kNearest(float[] testPredictor, int testPredictorOffset, float[] predictors, int predictorsStride, int[] outputs, int k){
		if(outputs.length > k){
			PriorityQueue<CategoryWithDistance> best = new PriorityQueue<>(k, new CategoryWithDistanceCompare());
			
			for(int i = 0; i < k; ++i){
				float distance = euclidianDistance(testPredictor, testPredictorOffset, predictors, i * predictorsStride, predictorsStride);
				best.add(new CategoryWithDistance(outputs[i], distance));
			}
			
			for(int i = k; i < outputs.length; ++i){
				float distance = euclidianDistance(testPredictor, testPredictorOffset, predictors, i * predictorsStride, predictorsStride);
				if(distance < best.peek().distance){
					best.poll();
					best.add(new CategoryWithDistance(outputs[i], distance));
				}
			}
			
			int[] kNearest = new int[k];
			int i = 0;
			for(CategoryWithDistance cat : best){
				kNearest[i] = cat.category;
				++i;
			}
			return kNearest;
		} else {
			return outputs;
		}
	}
	
	private static int mostCommon(int[] objects){
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int obj : objects) {
		    Integer count = map.get((Integer)obj);
		    map.put(obj, count != null ? count+1 : 0);
		}
		
		return Collections.max(map.entrySet(), (Entry<Integer, Integer> o1, Entry<Integer, Integer> o2) -> 
			o1.getValue() - (o2.getValue())).getKey();
	}
	
	private static void toByteArray(PriorityQueue<CategoryWithDistance> best, byte[] bytes){
		ByteBuffer buffer = ByteBuffer.wrap(bytes);
		buffer.putInt(best.size());
		for(CategoryWithDistance categoryWithDistance : best){
			buffer.putFloat(categoryWithDistance.distance);
			buffer.putInt(categoryWithDistance.category);
		}
	}
	
	private static PriorityQueue<CategoryWithDistance> fromByteArray(byte[] bytes){
		ByteBuffer buffer = ByteBuffer.wrap(bytes);
		int size = buffer.getInt();
		PriorityQueue<CategoryWithDistance> bestQueue = new PriorityQueue<CategoryWithDistance>(size, new CategoryWithDistanceCompare()); 
		for(int i = 0; i < size; ++i){
			float distance = buffer.getFloat();
			int category = buffer.getInt();
			bestQueue.add(new CategoryWithDistance(category, distance));
		}
		return bestQueue;
	}
	
	private static void combineBestNeighbours(PriorityQueue<CategoryWithDistance> bestIn, PriorityQueue<CategoryWithDistance> BestInOut,
			int k){
		
		int numberToTransfer = k - bestIn.size();
		for(int i = 0; i < numberToTransfer && !bestIn.isEmpty(); ++i){
			BestInOut.add(bestIn.poll());
		}
		for(CategoryWithDistance value : bestIn){
			if(value.distance < BestInOut.peek().distance){
				BestInOut.poll();
				BestInOut.add(value);
			}
		}
	}
	private interface ParallelForFunction {
		void run(int index);
	}
	private static void jompParallelFor(int start, int stop, int step, ParallelForFunction function) {
		//Sadly the JOMP Compiler is built around a Java 1.1 parser, which means it can't parse generics or lambdas etc.
		//Therefore I have used the functions in the runtime library directly.
		try {
			OMP.doParallel(new jomp.runtime.BusyTask(){
				@Override
				public void go(int __omp_me) throws Throwable {
					jomp.runtime.LoopData __omp_WholeData2 = new jomp.runtime.LoopData();
					jomp.runtime.LoopData __omp_ChunkData1 = new jomp.runtime.LoopData();
					__omp_WholeData2.start = start;
					__omp_WholeData2.stop = stop;
					__omp_WholeData2.step = 1;
					jomp.runtime.OMP.setChunkStatic(__omp_WholeData2);
					while(!__omp_ChunkData1.isLast && jomp.runtime.OMP.getLoopStatic(__omp_me, __omp_WholeData2, __omp_ChunkData1)) {
						for(;;){
							if(__omp_ChunkData1.stop > __omp_WholeData2.stop) __omp_ChunkData1.stop = __omp_WholeData2.stop;
							if(__omp_ChunkData1.start >= __omp_WholeData2.stop) break;
							for(int i = (int)__omp_ChunkData1.start; i < __omp_ChunkData1.stop; i += __omp_ChunkData1.step) {
								function.run(i);
							}
							if(__omp_ChunkData1.startStep == 0)
								break;
							__omp_ChunkData1.start += __omp_ChunkData1.startStep;
							__omp_ChunkData1.stop += __omp_ChunkData1.startStep;
						}
					}
					jomp.runtime.OMP.doBarrier(__omp_me);
				}
			});
		} catch(Throwable __omp_exception) {
			System.err.println("OMP Warning: Illegal thread exception ignored!");
			System.err.println(__omp_exception);
		}
	}
	
	private interface Implementation {
		void fit(float[] predictors, int predictorsStride, int[] outputs);
		int[] predict(float[] testPredictors);
	}
	
	private static class SingleThreadedImplementation implements Implementation {
		
		private int numberOfNeighbours;
		private float[] predictors;
		private int predictorsStride;
		private int[] outputs;
		
		public SingleThreadedImplementation(int numberOfNeighbours){
			this.numberOfNeighbours = numberOfNeighbours;
		}

		@Override
		public void fit(float[] predictors, int predictorsStride, int[] outputs) {
			this.predictors = predictors;
			this.predictorsStride = predictorsStride;
			this.outputs = outputs;
		}

		@Override
		public int[] predict(float[] testPredictors) {
			int[] predictions = new int[testPredictors.length / predictorsStride];
			for(int i = 0; i < predictions.length; ++i) {
				int[] kNearest = kNearest(testPredictors, i * predictorsStride, predictors, predictorsStride, outputs, numberOfNeighbours);
				predictions[i] = mostCommon(kNearest);
			}
			return predictions;
		}
		
	}
	
	private static class MultiThreadedImplementation implements Implementation {
		
		private int numberOfNeighbours;
		private float[] predictors;
		private int predictorsStride;
		private int[] outputs;
		
		public MultiThreadedImplementation(int numberOfNeighbours){
			this.numberOfNeighbours = numberOfNeighbours;
		}
		
		@Override
		public void fit(float[] predictors, int predictorsStride, int[] outputs) {
			this.predictors = predictors;
			this.predictorsStride = predictorsStride;
			this.outputs = outputs;
		}
		
		@Override
		public int[] predict(float[] testPredictors) {
			int[] predictions = new int[testPredictors.length / predictorsStride];
			jompParallelFor(0, predictions.length, 1, (i)->{
				int[] kNearest = kNearest(testPredictors, i * predictorsStride, predictors, predictorsStride, outputs, numberOfNeighbours);
				predictions[i] = mostCommon(kNearest);
			});
			return predictions;
		}
		
	}
	
	private static class MultiComputerMPJImplementation implements Implementation {

		protected int numberOfNeighbours;
		private float[] predictors;
		protected int predictorsStride;
		private int[] outputs;
		
		public MultiComputerMPJImplementation(int numberOfNeighbours){
			this.numberOfNeighbours = numberOfNeighbours;
		}
		
		@Override
		public void fit(float[] predictors, int predictorsStride, int[] outputs) {
			this.predictors = predictors;
			this.predictorsStride = predictorsStride;
			this.outputs = outputs;
		}

		@Override
		public int[] predict(float[] testPredictors) {
			int rank = MPI.COMM_WORLD.Rank();
			final int dataLength = 8 * numberOfNeighbours + 4;
			Op reduceOp = makeReduceOp();
			int length = testPredictors.length / predictorsStride;
			
			if(rank == 0) {
				int[] predictions = new int[length];
				for(int i = 0; i < length; ++i) {
					predictValueMaster(testPredictors, i, dataLength, reduceOp, predictions);
				}
				return predictions;
			} else {
				for(int i = 0; i < length; ++i) {
					predictValueSlave(testPredictors, i, dataLength, reduceOp);
				}
				return null;
			}
		}
		
		protected void predictValueMaster(float[] testPredictors, int i, int dataLength, Op reduceOp, int[] predictions) {
			PriorityQueue<CategoryWithDistance> kNearestLocal = kNearestByDistance(testPredictors, i * predictorsStride, predictors,
					predictorsStride, outputs, numberOfNeighbours);
			byte[] byteBest = new byte[dataLength];
			byte[] kNearestLocalBytes = new byte[dataLength];
			toByteArray(kNearestLocal, kNearestLocalBytes);
			MPI.COMM_WORLD.Reduce(kNearestLocalBytes, 0, byteBest, 0, dataLength, MPI.BYTE, reduceOp, 0);
			PriorityQueue<CategoryWithDistance> best = fromByteArray(byteBest);
			int[] kNearest = new int[best.size()];
			for(int j = 0; j < kNearest.length; ++j){
				kNearest[j] = best.poll().category;
			}
			predictions[i] = mostCommon(kNearest);
		}
		
		protected void predictValueSlave(float[] testPredictors, int i, int dataLength, Op reduceOp) {
			PriorityQueue<CategoryWithDistance> kNearestLocal = kNearestByDistance(testPredictors, i * predictorsStride, predictors, 
					predictorsStride, outputs, numberOfNeighbours);
			byte[] kNearestLocalBytes = new byte[dataLength];
			toByteArray(kNearestLocal, kNearestLocalBytes);
			byte[] byteBest = new byte[dataLength];
			MPI.COMM_WORLD.Reduce(kNearestLocalBytes, 0, byteBest, 0, dataLength, MPI.BYTE, reduceOp, 0);
		}
		
		protected static Op makeReduceOp() {
			return new Op(new mpi.User_function(){
				@Override
				public void Call(Object invec, int inoffset, Object inoutvec, int inoutoffset, int count, Datatype datatype){
					combineBestNeighboursByteArrays((byte[])invec, (byte[])inoutvec);
				}
			}, true);
		}
		
		private static void combineBestNeighboursByteArrays(byte[] bestIn, byte[] BestInOut){
			int k = (bestIn.length - 4) / 8;
			PriorityQueue<CategoryWithDistance> best1 = fromByteArray(bestIn);
			PriorityQueue<CategoryWithDistance> best2 = fromByteArray(BestInOut);
			combineBestNeighbours(best1, best2, k);
			toByteArray(best2, BestInOut);
		}
	}
	
	private class MultiThreadedJompMultiComputerMPJImplementation extends MultiComputerMPJImplementation
	{
		public MultiThreadedJompMultiComputerMPJImplementation(int numberOfNeighbours){
			super(numberOfNeighbours);
		}
		
		@Override
		public int[] predict(float[] testPredictors) {
			int rank = MPI.COMM_WORLD.Rank();
			final int dataLength = 8 * numberOfNeighbours + 4;
			Op reduceOp = makeReduceOp();
			int length = testPredictors.length / predictorsStride;
			
			if(rank == 0) {
				int[] predictions = new int[length];
				jompParallelFor(0, length, 1, (i)->{
					predictValueMaster(testPredictors, i, dataLength, reduceOp, predictions);
				});
				return predictions;
			} else {
				jompParallelFor(0, length, 1, (i)->{
					predictValueSlave(testPredictors, i, dataLength, reduceOp);
				});
				return null;
			}
		}
	}
	
	private Implementation implementation;
	
	public NearestNeighboursClassifier(int numberOfNeighbours, ExecutionPolicy executionPolicy) {
		switch(executionPolicy) {
		case singleThreaded:
			implementation = new SingleThreadedImplementation(numberOfNeighbours);
			break;
		case multiThreadedJomp:
			implementation = new MultiThreadedImplementation(numberOfNeighbours);
			break;
		case multiComputerMPJ:
			implementation = new MultiComputerMPJImplementation(numberOfNeighbours);
			break;
		case multiThreadedJompMultiComputerMPJ:
			implementation = new MultiThreadedJompMultiComputerMPJImplementation(numberOfNeighbours);
		}
	}
	
	public NearestNeighboursClassifier fit(float[] predictors, int predictorsStride, int[] outputs){
		implementation.fit(predictors, predictorsStride, outputs);
		return this;
	}
	
	public int[] predict(float[] testPredictors) {
		return implementation.predict(testPredictors);
	}
	
}
