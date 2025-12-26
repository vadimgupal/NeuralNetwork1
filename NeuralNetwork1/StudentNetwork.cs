using System;
using System.Diagnostics;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        // --------- Параметры и внутреннее состояние сети ---------

        private readonly int[] structure;   // Кол-во нейронов на каждом слое (включая вход и выход)
        private readonly int layersCount;   // Число слоёв

        // Активности нейронов по слоям: neurons[слой][нейрон]
        private readonly double[][] neurons;

        // Смещения (bias) по слоям, кроме входного: biases[слой][нейрон]
        private readonly double[][] biases;

        // Весовые матрицы: weights[l] – матрица связей между слоями (l-1) и l
        // Размер weights[l]: [structure[l-1], structure[l]]
        private readonly double[][,] weights;

        // Δ для backprop: deltas[l][нейрон]
        private readonly double[][] deltas;

        // Источник случайных чисел для инициализации
        private readonly Random rnd = new Random();

        // Скорость обучения
        private readonly double learningRate = 0.01;

        // Секундомер (аналогично AccordNet)
        public Stopwatch stopWatch = new Stopwatch();

        public StudentNetwork(int[] structure)
        {
            if (structure == null || structure.Length < 2)
                throw new ArgumentException("Структура сети должна содержать минимум входной и выходной слои");

            this.structure = (int[])structure.Clone();
            layersCount = structure.Length;

            neurons = new double[layersCount][];
            biases = new double[layersCount][];
            weights = new double[layersCount][,];
            deltas = new double[layersCount][];

            // Инициализация слоёв
            for (int l = 0; l < layersCount; l++)
            {
                neurons[l] = new double[this.structure[l]];
                deltas[l] = new double[this.structure[l]];

                if (l == 0)
                {
                    // Входной слой – без весов и bias
                    biases[l] = null;
                    weights[l] = null;
                }
                else
                {
                    biases[l] = new double[this.structure[l]];
                    weights[l] = new double[this.structure[l - 1], this.structure[l]];

                    // Инициализация весов небольшими случайными значениями
                    double scale = 2.0 / Math.Sqrt(this.structure[l - 1]); // типа Xavier
                    for (int i = 0; i < this.structure[l - 1]; i++)
                        for (int j = 0; j < this.structure[l]; j++)
                            weights[l][i, j] = (rnd.NextDouble() - 0.5) * scale;

                    // Смещения можно оставить нулевыми
                    for (int j = 0; j < this.structure[l]; j++)
                        biases[l][j] = 0.0;
                }
            }
        }

        // ------------------- Публичные методы обучения -------------------

        /// <summary>
        /// Обучение сети одному образу до достижения acceptableError
        /// </summary>
        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            // Здесь parallel игнорируем – учим последовательно один образ
            int iterations = 0;
            double error = double.PositiveInfinity;

            const int maxIterations = 100000; // На всякий случай, чтобы не зациклиться

            while (error > acceptableError && iterations < maxIterations)
            {
                iterations++;

                // Прямой проход
                var output = Compute(sample.input);

                // Обновляем в sample вектор ошибок и распознанный класс
                sample.ProcessPrediction(output);
                error = sample.EstimatedError();

                // Обратное распространение ошибки
                Backpropagate(sample);
            }

            return iterations;
        }

        /// <summary>
        /// Обучение сети на датасете
        /// </summary>
        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            // Для простоты training делаем последовательным даже при parallel = true
            // (распараллеливание – опциональное улучшение)

            stopWatch.Restart();

            double datasetError = double.PositiveInfinity;
            int epoch = 0;

            while (epoch < epochsCount && datasetError > acceptableError)
            {
                epoch++;
                double sumError = 0.0;

                foreach (Sample sample in samplesSet)
                {
                    // Прямой проход
                    var output = Compute(sample.input);

                    // Обновляем в sample выходы и вектор ошибок
                    sample.ProcessPrediction(output);
                    sumError += sample.EstimatedError();

                    // Обратное распространение и обновление весов
                    Backpropagate(sample);
                }

                datasetError = sumError / samplesSet.Count;

                // Сообщаем форме о прогрессе
                OnTrainProgress(
                    progress: (epoch * 1.0) / epochsCount,
                    error: datasetError,
                    time: stopWatch.Elapsed
                );
            }

            stopWatch.Stop();

            // Финальный вызов – "мы закончили"
            OnTrainProgress(1.0, datasetError, stopWatch.Elapsed);

            return datasetError;
        }


        protected override double[] Compute(double[] input)
        {
            if (input.Length != structure[0])
                throw new ArgumentException("Размер входного вектора не совпадает с размером входного слоя сети");

            // Заполняем входной слой
            for (int i = 0; i < structure[0]; i++)
                neurons[0][i] = input[i];

            // Прямое распространение по всем последующим слоям
            for (int l = 1; l < layersCount; l++)
            {
                int prevSize = structure[l - 1];
                int curSize = structure[l];

                for (int j = 0; j < curSize; j++)
                {
                    double sum = biases[l][j];

                    for (int i = 0; i < prevSize; i++)
                        sum += neurons[l - 1][i] * weights[l][i, j];

                    neurons[l][j] = Sigmoid(sum);
                }
            }

            // Возвращаем копию выходного слоя
            int last = layersCount - 1;
            double[] result = new double[structure[last]];
            Array.Copy(neurons[last], result, structure[last]);
            return result;
        }

        /// <summary>
        /// Обратное распространение ошибки и обновление весов
        /// </summary>
        private void Backpropagate(Sample sample)
        {
            int last = layersCount - 1;

            // 1. Вычисляем дельта для выходного слоя
            for (int j = 0; j < structure[last]; j++)
            {
                double a = neurons[last][j];   // активация
                double y = (j == (int)sample.actualClass) ? 1.0 : 0.0; // целевой выход
                double dC_da = (a - y);        // производная квадратичной ошибки по a_j
                deltas[last][j] = dC_da * a * (1.0 - a); // * сигмоид'(net)
            }

            // 2. дельта для скрытых слоёв (идём назад)
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

            // 3. Обновляем веса и смещения (градиентный спуск)
            for (int l = 1; l < layersCount; l++)
            {
                int prevSize = structure[l - 1];
                int curSize = structure[l];

                // Веса
                for (int i = 0; i < prevSize; i++)
                    for (int j = 0; j < curSize; j++)
                        weights[l][i, j] -= learningRate * neurons[l - 1][i] * deltas[l][j];

                // Смещения
                for (int j = 0; j < curSize; j++)
                    biases[l][j] -= learningRate * deltas[l][j];
            }
        }

        // ------------------- Вспомогательные функции -------------------

        private static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
    }
}
