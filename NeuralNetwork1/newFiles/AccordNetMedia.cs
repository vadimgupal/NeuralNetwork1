using System.Diagnostics;
using System.IO;
using System.Linq;
using Accord.Neuro;
using Accord.Neuro.Learning;

namespace NeuralNetwork1
{
    class AccordNetMedia : BaseNetworkMedia
    {
        private ActivationNetwork network;
        public Stopwatch stopWatch = new Stopwatch();

        public AccordNetMedia(int[] structure)
        {
            network = new ActivationNetwork(new SigmoidFunction(2.0), structure[0], structure.Skip(1).ToArray());
            new NguyenWidrow(network).Randomize();
        }

        public override int Train(MediaSample sample, double acceptableError, bool parallel)
        {
            var teacher = MakeTeacher(parallel);

            int iters = 1;
            while (teacher.Run(sample.input, sample.Target) > acceptableError)
                ++iters;

            return iters;
        }

        private ISupervisedLearning MakeTeacher(bool parallel)
        {
            if (parallel)
                return new ParallelResilientBackpropagationLearning(network);
            return new ResilientBackpropagationLearning(network);
        }

        public override double TrainOnDataSet(MediaSamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            double[][] inputs = new double[samplesSet.Count][];
            double[][] outputs = new double[samplesSet.Count][];

            int idx = 0;
            foreach (MediaSample s in samplesSet.samples)
            {
                inputs[idx] = s.input;
                outputs[idx] = s.Target;
                idx++;
            }

            int epoch = 0;
            var teacher = MakeTeacher(parallel);
            double error = double.PositiveInfinity;

#if DEBUG
            StreamWriter errorsFile = File.CreateText("errors_media.csv");
#endif

            stopWatch.Restart();

            while (epoch < epochsCount && error > acceptableError)
            {
                epoch++;
                error = teacher.RunEpoch(inputs, outputs);

#if DEBUG
                errorsFile.WriteLine(error);
#endif
                OnTrainProgress((epoch * 1.0) / epochsCount, error, stopWatch.Elapsed);
            }

#if DEBUG
            errorsFile.Close();
#endif

            OnTrainProgress(1.0, error, stopWatch.Elapsed);
            stopWatch.Stop();
            return error;
        }

        protected override double[] Compute(double[] input)
        {
            return network.Compute(input);
        }
    }
}
