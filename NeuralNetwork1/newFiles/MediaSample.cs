using System;

namespace NeuralNetwork1
{
    public class MediaSample
    {
        public double[] input;
        public double[] error;
        public MediaSymbol actualClass;
        public MediaSymbol recognizedClass;

        //целевой вектор
        public double[] Target { get; private set; }

        // Реальный выход сети
        public double[] Output { get; private set; }

        public MediaSample(double[] inputValues, int classesCount, MediaSymbol sampleClass)
        {
            input = (double[])inputValues.Clone();

            Target = new double[classesCount];
            if (sampleClass != MediaSymbol.Undef)
                Target[(int)sampleClass] = 1.0;

            Output = new double[classesCount];

            actualClass = sampleClass;
            recognizedClass = 0;
            error = new double[classesCount];
        }

        public MediaSymbol ProcessPrediction(double[] neuralOutput)
        {
            Output = neuralOutput;

            //просто argmax (без привязки к actualClass)
            int best = 0;
            for (int i = 1; i < Output.Length; i++)
                if (Output[i] > Output[best])
                    best = i;

            recognizedClass = (MediaSymbol)best;

            //ошибку считаем ТОЛЬКО если есть метка
            if (actualClass != MediaSymbol.Undef)
            {
                for (int i = 0; i < Output.Length; i++)
                    error[i] = Output[i] - Target[i];
            }
            else
            {
                Array.Clear(error, 0, error.Length);
            }

            return recognizedClass;
        }

        public double EstimatedError()
        {
            double sum = 0;
            for (int i = 0; i < Output.Length; i++)
                sum += error[i] * error[i];
            return sum;
        }

        public bool Correct()
        {
            return actualClass != MediaSymbol.Undef && actualClass == recognizedClass;
        }
    }
}
