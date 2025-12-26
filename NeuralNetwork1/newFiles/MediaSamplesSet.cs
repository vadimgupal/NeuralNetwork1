using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace NeuralNetwork1
{
    public class MediaSamplesSet : IEnumerable
    {
        public List<MediaSample> samples = new List<MediaSample>();

        public int Count { get { return samples.Count; } }

        public void AddSample(MediaSample s)
        {
            samples.Add(s);
        }

        public IEnumerator GetEnumerator()
        {
            return samples.GetEnumerator();
        }

        public double TestNetwork(BaseNetworkMedia network)
        {
            double correct = 0;
            double wrong = 0;

            foreach (var s in samples)
            {
                if (s.actualClass == network.Predict(s)) correct++;
                else wrong++;
            }

            return correct / (correct + wrong);
        }

        public void Save(string path)
        {
            using (var sw = new StreamWriter(path, false))
            {
                // формат:
                // classIndex;v0;v1;...;v1023
                foreach (var s in samples)
                {
                    sw.Write((int)s.actualClass);

                    for (int i = 0; i < s.input.Length; i++)
                    {
                        sw.Write(';');
                        sw.Write(s.input[i].ToString(CultureInfo.InvariantCulture));
                    }

                    sw.WriteLine();
                }
            }
        }

        public static MediaSamplesSet Load(string path)
        {
            var set = new MediaSamplesSet();

            foreach (var line in File.ReadLines(path))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                var parts = line.Split(';');
                int classIndex = int.Parse(parts[0]);

                double[] input = new double[parts.Length - 1];
                for (int i = 1; i < parts.Length; i++)
                    input[i - 1] = double.Parse(parts[i], CultureInfo.InvariantCulture);

                var sample = new MediaSample(input, Enum.GetValues(typeof(MediaSymbol)).Length - 1, (MediaSymbol)classIndex);
                set.AddSample(sample);
            }

            return set;
        }
    }
}
