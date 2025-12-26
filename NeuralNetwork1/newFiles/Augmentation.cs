using System;
using System.Drawing;

namespace NeuralNetwork1
{
    public static class Augmentation
    {
        private static readonly Random Rnd = new Random();

        // Добавим "соль-перец" (часть пикселей случайно инвертируем)
        public static Bitmap AddSaltPepper(Bitmap binary, double probability)
        {
            Bitmap dst = (Bitmap)binary.Clone();

            for (int y = 0; y < dst.Height; y++)
                for (int x = 0; x < dst.Width; x++)
                {
                    if (Rnd.NextDouble() < probability)
                    {
                        bool isBlack = dst.GetPixel(x, y).R == 0;
                        dst.SetPixel(x, y, isBlack ? Color.White : Color.Black);
                    }
                }

            return dst;
        }

        // Лёгкий сдвиг внутри кадра (для устойчивости)
        public static Bitmap Shift(Bitmap src, int dx, int dy)
        {
            Bitmap dst = new Bitmap(src.Width, src.Height);
            using (Graphics g = Graphics.FromImage(dst))
            {
                g.Clear(Color.White);
                g.DrawImage(src, dx, dy);
            }
            return dst;
        }
    }
}
