using System;
using System.Diagnostics;

namespace NeuralNetwork1
{
    public class StudentNetworkMedia : BaseNetworkMedia
    {
        private readonly int[] structure;
        private readonly int layersCount;

        private readonly double[][] neurons;
        private readonly double[][] biases;
        private readonly double[][,] weights;
        private readonly double[][] deltas;

        private readonly Random rnd = new Random();
        private readonly double learningRate = 0.01;

        public Stopwatch stopWatch = new Stopwatch();

        public StudentNetworkMedia(int[] structure)
        {
            if (structure == null || structure.Length < 2)
                throw new ArgumentException("Структура сети должна содержать минимум входной и выходной слои");

            this.structure = (int[])structure.Clone();
            layersCount = this.structure.Length;

            neurons = new double[layersCount][];
            biases = new double[layersCount][];
            weights = new double[layersCount][,];
            deltas = new double[layersCount][];

            for (int l = 0; l < layersCount; l++)
            {
                neurons[l] = new double[this.structure[l]];
                deltas[l] = new double[this.structure[l]];

                if (l == 0)
                {
                    biases[l] = null;
                    weights[l] = null;
                }
                else
                {
                    biases[l] = new double[this.structure[l]];
                    weights[l] = new double[this.structure[l - 1], this.structure[l]];

                    double scale = 2.0 / Math.Sqrt(this.structure[l - 1]);
                    for (int i = 0; i < this.structure[l - 1]; i++)
                        for (int j = 0; j < this.structure[l]; j++)
                            weights[l][i, j] = (rnd.NextDouble() - 0.5) * scale;

                    for (int j = 0; j < this.structure[l]; j++)
                        biases[l][j] = 0.0;
                }
            }
        }

        public override int Train(MediaSample sample, double acceptableError, bool parallel)
        {
            int iterations = 0;
            double error = double.PositiveInfinity;

            const int maxIterations = 100000;

            while (error > acceptableError && iterations < maxIterations)
            {
                iterations++;

                double[] output = Compute(sample.input);
                sample.ProcessPrediction(output);
                error = sample.EstimatedError();

                Backpropagate(sample);
            }

            return iterations;
        }

        public override double TrainOnDataSet(MediaSamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            stopWatch.Restart();

            double datasetError = double.PositiveInfinity;
            int epoch = 0;

            while (epoch < epochsCount && datasetError > acceptableError)
            {
                epoch++;
                double sumError = 0.0;

                var list = samplesSet.samples;
                for (int i = list.Count - 1; i > 0; i--)
                {
                    int j = rnd.Next(i + 1);
                    var tmp = list[i];
                    list[i] = list[j];
                    list[j] = tmp;
                }

                foreach (MediaSample sample in samplesSet.samples)
                {
                    double[] output = Compute(sample.input);
                    sample.ProcessPrediction(output);
                    sumError += sample.EstimatedError();

                    Backpropagate(sample);
                }

                datasetError = sumError / samplesSet.Count;

                OnTrainProgress((epoch * 1.0) / epochsCount, datasetError, stopWatch.Elapsed);
            }

            stopWatch.Stop();
            OnTrainProgress(1.0, datasetError, stopWatch.Elapsed);

            return datasetError;
        }

        protected override double[] Compute(double[] input)
        {
            if (input.Length != structure[0])
                throw new ArgumentException("Размер входного вектора не совпадает с размером входного слоя");

            for (int i = 0; i < structure[0]; i++)
                neurons[0][i] = input[i];

            for (int l = 1; l < layersCount; l++)
            {
                int prevSize = structure[l - 1];
                int curSize = structure[l];

                for (int j = 0; j < curSize; j++)
                {
                    double sum = biases[l][j];
                    for (int i = 0; i < prevSize; i++)
                        sum += neurons[l - 1][i] * weights[l][i, j];

                    // скрытые слои — sigmoid
                    if (l != layersCount - 1)
                        neurons[l][j] = Sigmoid(sum);
                    else
                        neurons[l][j] = sum; // logits
                }

                // на выходе — softmax
                if (l == layersCount - 1)
                    SoftmaxInPlace(neurons[l]);
            }

            int last = layersCount - 1;
            double[] result = new double[structure[last]];
            Array.Copy(neurons[last], result, structure[last]);
            return result;
        }

        private void Backpropagate(MediaSample sample)
        {
            int last = layersCount - 1;

            // Выходной слой
            for (int j = 0; j < structure[last]; j++)
            {
                // если образ "без метки" (Undef), то не учим
                if (sample.actualClass == MediaSymbol.Undef)
                {
                    deltas[last][j] = 0;
                    continue;
                }

                double a = neurons[last][j];
                double y = (j == (int)sample.actualClass) ? 1.0 : 0.0;
                
                deltas[last][j] = (a - y);
            }

            // Скрытые слои
            for (int l = last - 1; l >= 1; l--)
            {
                int curSize = structure[l];
                int nextSize = structure[l + 1];

                for (int i = 0; i < curSize; i++)
                {
                    double sum = 0.0;
                    for (int j = 0; j < nextSize; j++)
                        sum += weights[l + 1][i, j] * deltas[l + 1][j];

                    double a = neurons[l][i];
                    deltas[l][i] = sum * a * (1.0 - a);
                }
            }

            // Обновление весов и bias
            for (int l = 1; l < layersCount; l++)
            {
                int prevSize = structure[l - 1];
                int curSize = structure[l];

                for (int i = 0; i < prevSize; i++)
                    for (int j = 0; j < curSize; j++)
                        weights[l][i, j] -= learningRate * neurons[l - 1][i] * deltas[l][j];

                for (int j = 0; j < curSize; j++)
                    biases[l][j] -= learningRate * deltas[l][j];
            }
        }

        private static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private static void SoftmaxInPlace(double[] v)
        {
            double max = v[0];
            for (int i = 1; i < v.Length; i++)
                if (v[i] > max) max = v[i];

            double sum = 0.0;
            for (int i = 0; i < v.Length; i++)
            {
                v[i] = Math.Exp(v[i] - max);
                sum += v[i];
            }

            for (int i = 0; i < v.Length; i++)
                v[i] /= sum;
        }

    }
}
