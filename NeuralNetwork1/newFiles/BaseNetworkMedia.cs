using System;

namespace NeuralNetwork1
{
    public abstract class BaseNetworkMedia
    {
        public event TrainProgressHandler TrainProgress;

        public abstract int Train(MediaSample sample, double acceptableError, bool parallel);
        public abstract double TrainOnDataSet(MediaSamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel);

        protected abstract double[] Compute(double[] input);

        public MediaSymbol Predict(MediaSample sample)
        {
            return sample.ProcessPrediction(Compute(sample.input));
        }

        protected virtual void OnTrainProgress(double progress, double error, TimeSpan time)
        {
            if (TrainProgress != null)
                TrainProgress(progress, error, time);
        }
    }
}
