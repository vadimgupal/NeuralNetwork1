using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetwork1
{
    public partial class NeuralNetworksStand : Form
    {
        /// <summary>
        /// Генератор изображений (образов)
        /// </summary>
        //GenerateImage generator = new GenerateImage();

        private CameraCapture camera = new CameraCapture();
        private MediaSamplesSet trainSamples = new MediaSamplesSet();

        private const int ImgSize = 32;      // 32x32
        private const byte Threshold = 170;  // порог бинаризации
        private const int Padding = 5;       // поля при обрезке

        /// <summary>
        /// Текущая выбранная через селектор нейросеть
        /// </summary>
        //public BaseNetwork Net
        //{
        //    get
        //    {
        //        var selectedItem = (string) netTypeBox.SelectedItem;
        //        if (!networksCache.ContainsKey(selectedItem))
        //            networksCache.Add(selectedItem, CreateNetwork(selectedItem));

        //        return networksCache[selectedItem];
        //    }
        //}

        public BaseNetworkMedia Net
        {
            get
            {
                var selectedItem = (string)netTypeBox.SelectedItem;
                if (!networksCache.ContainsKey(selectedItem))
                    networksCache.Add(selectedItem, CreateNetwork(selectedItem));
                return networksCache[selectedItem];
            }
        }

        private readonly Dictionary<string, Func<int[], BaseNetworkMedia>> networksFabric;
        private Dictionary<string, BaseNetworkMedia> networksCache = new Dictionary<string, BaseNetworkMedia>();


        /// <summary>
        /// Конструктор формы стенда для работы с сетями
        /// </summary>
        /// <param name="networksFabric">Словарь функций, создающих сети с заданной структурой</param>
        public NeuralNetworksStand(Dictionary<string, Func<int[], BaseNetworkMedia>> networksFabric)
        {
            InitializeComponent();
            this.networksFabric = networksFabric;

            netTypeBox.Items.AddRange(this.networksFabric.Keys.Select(s => (object)s).ToArray());
            netTypeBox.SelectedIndex = 0;

            // символы
            symbolBox.Items.Clear();
            symbolBox.Items.Add("Play (▶)");
            symbolBox.Items.Add("Stop (■)");
            symbolBox.Items.Add("Pause (||)");
            symbolBox.Items.Add("Rewind (<<)");
            symbolBox.Items.Add("Forward (>>)");
            symbolBox.SelectedIndex = 0;

            classCounter.Value = 5;
            classCounter.Enabled = false;

            // структура сети под 32×32 => 1024 входа
            netStructureBox.Text = "1024;128;64;5";

            // создаём сети (как раньше, через кнопку пересоздания)
            button3_Click(this, null);

            // старт камеры
            camera.FrameArrived += OnCameraFrame;
            camera.Start(0);
        }

        private void OnCameraFrame(Bitmap frame)
        {
            // Важно: в WinForms нельзя трогать UI из потока камеры
            if (pictureBox1.InvokeRequired)
            {
                pictureBox1.BeginInvoke(new Action(() => OnCameraFrame(frame)));
                return;
            }

            // Показываем live кадр
            if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
            pictureBox1.Image = (Bitmap)frame.Clone();

            frame.Dispose(); // освобождаем кадр, который пришёл из camera
        }

        public void UpdateLearningInfo(double progress, double error, TimeSpan elapsedTime)
        {
            if (progressBar1.InvokeRequired)
            {
                progressBar1.BeginInvoke(
                    new TrainProgressHandler(UpdateLearningInfo),
                    progress, error, elapsedTime
                );
                return;
            }


            StatusLabel.Text = "Ошибка: " + error;
            int progressPercent = (int) Math.Round(progress * 100);
            progressPercent = Math.Min(100, Math.Max(0, progressPercent));
            elapsedTimeLabel.Text = "Затраченное время : " + elapsedTime.Duration().ToString(@"hh\:mm\:ss\:ff");
            progressBar1.Value = progressPercent;
        }


        private void set_result(MediaSample sample)
        {
            label1.ForeColor = Color.Black;

            label1.Text = "Распознано: " + sample.recognizedClass;

            label8.Text = string.Join("\n", sample.Output.Select(d => d.ToString(CultureInfo.InvariantCulture)));
        }


        private void pictureBox1_MouseClick(object sender, MouseEventArgs e)
        {
            if (Net == null) return;

            Bitmap frame = null;
            try
            {
                frame = camera.Snapshot();

                double[] input = ImagePreprocessor.PreprocessToVectorSmart(frame, ImgSize, Threshold, Padding);
                double sum = input.Sum();
                StatusLabel.Text = $"predict sum={sum:F0}";

                if (sum < 10) // порог подбери, но обычно 5..50
                {
                    label1.Text = "Не вижу символ (слишком мало черных пикселей).";
                    label1.ForeColor = Color.Red;
                    return;
                }
                int classesCount = (int)classCounter.Value;

                // для распознавания нам не важно actualClass — но MediaSample требует его
                // поэтому создаём "фиктивно", а подсветку Correct() можно игнорировать
                MediaSample sample = new MediaSample(input, classesCount, MediaSymbol.Undef);

                Net.Predict(sample);
                set_result(sample);
            }
            catch (Exception ex)
            {
                label1.Text = "Ошибка: " + ex.Message;
                label1.ForeColor = Color.Red;
            }
            finally
            {
                if (frame != null) frame.Dispose();
            }
        }

        private async Task<double> train_networkAsync(int epoches, double acceptable_error, bool parallel)
        {
            if (trainSamples.Count == 0)
            {
                label1.Text = "Сначала добавьте обучающие примеры!";
                label1.ForeColor = Color.Red;
                return 0;
            }

            // Блокируем UI как раньше
            label1.Text = "Выполняется обучение...";
            label1.ForeColor = Color.Red;
            groupBox1.Enabled = false;
            pictureBox1.Enabled = false;
            
            try
            {
                var curNet = Net;
                double f = await Task.Run(() => curNet.TrainOnDataSet(trainSamples, epoches, acceptable_error, parallel));

                label1.Text = "Готово. Клик по картинке = распознавание";
                label1.ForeColor = Color.Green;

                StatusLabel.Text = "Ошибка: " + f;
                StatusLabel.ForeColor = Color.Green;
                return f;
            }
            catch (Exception ex)
            {
                var msg = ex.Message;

                if (ex is AggregateException ae)
                    msg = string.Join(" | ", ae.Flatten().InnerExceptions.Select(e => e.Message));

                label1.Text = "Исключение: " + msg;
                label1.ForeColor = Color.Red;
                return 0;
            }
            finally
            {
                groupBox1.Enabled = true;
                pictureBox1.Enabled = true;
            }
        }


        //        private void button1_Click(object sender, EventArgs e)
        //        {
        //#pragma warning disable CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
        //            train_networkAsync((int) TrainingSizeCounter.Value, (int) EpochesCounter.Value,
        //                (100 - AccuracyCounter.Value) / 100.0, parallelCheckBox.Checked);
        //#pragma warning restore CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
        //        }

        private async void button1_Click(object sender, EventArgs e)
        {
            await train_networkAsync(
                (int)EpochesCounter.Value,
                (100 - AccuracyCounter.Value) / 100.0,
                parallelCheckBox.Checked
            );
        }


        private void button2_Click(object sender, EventArgs e)
        {
            if (trainSamples.Count == 0) return;
            double acc = trainSamples.TestNetwork(Net);
            StatusLabel.Text = $"Точность (на собранных примерах): {acc * 100:F2}%";
        }


        private void button3_Click(object sender, EventArgs e)
        {
            int[] structure = CurrentNetworkStructure();
            int classesCount = (int)classCounter.Value;

            if (structure.Length < 2 || structure[structure.Length - 1] != classesCount)
            {
                MessageBox.Show(
                    $"В сети должно быть минимум 2 слоя. Последний слой должен быть {classesCount}.",
                    "Ошибка", MessageBoxButtons.OK);
                return;
            }

            foreach (var network in networksCache.Values)
                network.TrainProgress -= UpdateLearningInfo;

            networksCache = networksCache.ToDictionary(oldNet => oldNet.Key, oldNet => CreateNetwork(oldNet.Key));
        }


        private int[] CurrentNetworkStructure()
        {
            return netStructureBox.Text.Split(';').Select(int.Parse).ToArray();
        }

        //private void classCounter_ValueChanged(object sender, EventArgs e)
        //{
        //    generator.FigureCount = (int) classCounter.Value;
        //    var vals = netStructureBox.Text.Split(';');
        //    if (!int.TryParse(vals.Last(), out _)) return;
        //    vals[vals.Length - 1] = classCounter.Value.ToString();
        //    netStructureBox.Text = vals.Aggregate((partialPhrase, word) => $"{partialPhrase};{word}");
        //}

        //private async void btnTrainOne_Click(object sender, EventArgs e)
        //{
        //    if (Net == null) return;

        //    // Блокируем UI
        //    groupBox1.Enabled = false;
        //    pictureBox1.Enabled = false;
        //    trainOneButton.Enabled = false;
        //    label1.Text = "Обучение по одному образцу...";
        //    label1.ForeColor = Color.Red;

        //    Sample fig = generator.GenerateFigure();
        //    pictureBox1.Image = generator.GenBitmap();
        //    pictureBox1.Invalidate();

        //    try
        //    {
        //        var curNet = Net;
        //        await Task.Run(() =>
        //        {
        //            // ВАЖНО: этот вызов больше не в UI-потоке
        //            curNet.Train(fig, 0.00005, parallelCheckBox.Checked);
        //        });

        //        set_result(fig);

        //        label1.Text = "Готово. Щёлкни по картинке для теста.";
        //        label1.ForeColor = Color.Green;
        //    }
        //    catch (Exception ex)
        //    {
        //        label1.Text = $"Исключение: {ex.Message}";
        //        label1.ForeColor = Color.Red;
        //    }
        //    finally
        //    {
        //        groupBox1.Enabled = true;
        //        pictureBox1.Enabled = true;
        //        trainOneButton.Enabled = true;
        //    }
        //}


        private BaseNetworkMedia CreateNetwork(string networkName)
        {
            var network = networksFabric[networkName](CurrentNetworkStructure());
            network.TrainProgress += UpdateLearningInfo;
            return network;
        }

        private void recreateNetButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Заново пересоздаёт сеть с указанными параметрами";
        }

        private void netTrainButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Обучить нейросеть с указанными параметрами";
        }

        private void testNetButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Тестировать нейросеть на тестовой выборке такого же размера";
        }

        private void NeuralNetworksStand_Load(object sender, EventArgs e)
        {

        }

        private void LoadDatasetFromFolder(string rootDir)
        {
            trainSamples = new MediaSamplesSet();

            int classesCount = (int)classCounter.Value;

            // Папки вида "00_Play", "01_Stop", ...
            var classDirs = System.IO.Directory.GetDirectories(rootDir);

            foreach (var dir in classDirs)
            {
                var folder = System.IO.Path.GetFileName(dir);

                // classId берем из первых двух символов "00", "01"...
                if (!int.TryParse(folder.Substring(0, 2), out int classId))
                    continue;

                var files = System.IO.Directory.GetFiles(dir, "*.png");

                foreach (var f in files)
                {
                    using (var bmp = (Bitmap)Bitmap.FromFile(f))
                    {
                        double[] input = ImagePreprocessor.PreprocessToVectorSmart(bmp, ImgSize, Threshold, Padding);
                        var sample = new MediaSample(input, classesCount, (MediaSymbol)classId);
                        trainSamples.AddSample(sample);
                    }
                }
            }

            StatusLabel.Text = $"Загружено из папки: {trainSamples.Count}";
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            try
            {
                camera.Dispose(); // остановить камеру и освободить ресурсы
            }
            catch { }

            base.OnFormClosing(e);
        }

        private void button1_Click_1(object sender, EventArgs e)
        {
            trainSamples.Save("train_media.csv");
            StatusLabel.Text = "Сохранено train_media.csv";
        }

        private void button2_Click_1(object sender, EventArgs e)
        {
            trainSamples = MediaSamplesSet.Load("train_media.csv");
            StatusLabel.Text = "Загружено: " + trainSamples.Count;
        }

        private void button3_Click_1(object sender, EventArgs e)
        {
            using (var dlg = new FolderBrowserDialog())
            {
                if (dlg.ShowDialog() == DialogResult.OK)
                {
                    LoadDatasetFromFolder(dlg.SelectedPath);
                    var grp = trainSamples.samples
    .GroupBy(s => s.actualClass)
    .Select(g => $"{g.Key}:{g.Count()}");

                    StatusLabel.Text = $"Загружено: {trainSamples.Count} | " + string.Join(" ", grp);
                }
            }
        }
    }
}