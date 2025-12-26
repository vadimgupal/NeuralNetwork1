using System;
using System.Drawing;
using Accord.Video;
using Accord.Video.DirectShow;

namespace NeuralNetwork1
{
    public sealed class CameraCapture : IDisposable
    {
        private VideoCaptureDevice _device;  
        private Bitmap _lastFrame;            

        public event Action<Bitmap> FrameArrived;

        public string[] GetCameraNames()
        {
            var devices = new FilterInfoCollection(FilterCategory.VideoInputDevice);
            var names = new string[devices.Count];
            for (int i = 0; i < devices.Count; i++) names[i] = devices[i].Name;
            return names;
        }

        public void Start(int cameraIndex = 0)
        {
            var devices = new FilterInfoCollection(FilterCategory.VideoInputDevice);
            if (devices.Count == 0) throw new InvalidOperationException("Камера не найдена.");
            if (cameraIndex < 0 || cameraIndex >= devices.Count) cameraIndex = 0;

            _device = new VideoCaptureDevice(devices[cameraIndex].MonikerString);
            _device.NewFrame += OnNewFrame;
            _device.Start();
        }

        private void OnNewFrame(object sender, NewFrameEventArgs eventArgs)
        {
            // Копируем кадр, чтобы не держать ссылку на внутренний буфер
            var bmp = (Bitmap)eventArgs.Frame.Clone();

            lock (this)
            {
                if (_lastFrame != null) _lastFrame.Dispose();
                _lastFrame = (Bitmap)bmp.Clone();
            }

            var handler = FrameArrived;
            if (handler != null)
                handler(bmp);
            else
                bmp.Dispose(); // чтобы не текла память, если никто не подписан
        }

        public Bitmap Snapshot()
        {
            lock (this)
            {
                if (_lastFrame == null)
                    throw new InvalidOperationException("Кадр ещё не получен.");
                return (Bitmap)_lastFrame.Clone();
            }
        }

        public void Dispose()
        {
            if (_device != null)
            {
                _device.NewFrame -= OnNewFrame;
                if (_device.IsRunning) _device.SignalToStop();
                _device = null;
            }

            if (_lastFrame != null)
            {
                _lastFrame.Dispose();
                _lastFrame = null;
            }
        }
    }
}
