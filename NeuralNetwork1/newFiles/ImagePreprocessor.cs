using System;
using System.Drawing;
using System.Drawing.Drawing2D;

namespace NeuralNetwork1
{
    public static class ImagePreprocessor
    {
        public static Bitmap ToBinary(Bitmap src, byte threshold)
        {
            Bitmap dst = new Bitmap(src.Width, src.Height);

            for (int y = 0; y < src.Height; y++)
                for (int x = 0; x < src.Width; x++)
                {
                    Color c = src.GetPixel(x, y);
                    byte g = (byte)(c.R * 0.299 + c.G * 0.587 + c.B * 0.114);
                    dst.SetPixel(x, y, g < threshold ? Color.Black : Color.White);
                }

            return dst;
        }

        public static Rectangle FindContentBounds(Bitmap bin, int padding)
        {
            int minX = bin.Width, minY = bin.Height, maxX = -1, maxY = -1;

            for (int y = 0; y < bin.Height; y++)
                for (int x = 0; x < bin.Width; x++)
                {
                    // чёрный пиксель
                    if (bin.GetPixel(x, y).R == 0)
                    {
                        if (x < minX) minX = x;
                        if (y < minY) minY = y;
                        if (x > maxX) maxX = x;
                        if (y > maxY) maxY = y;
                    }
                }

            // если чёрных пикселей нет — возвращаем всё изображение
            if (maxX < 0)
                return new Rectangle(0, 0, bin.Width, bin.Height);

            minX = Math.Max(0, minX - padding);
            minY = Math.Max(0, minY - padding);
            maxX = Math.Min(bin.Width - 1, maxX + padding);
            maxY = Math.Min(bin.Height - 1, maxY + padding);

            return Rectangle.FromLTRB(minX, minY, maxX + 1, maxY + 1);
        }

        public static Bitmap Crop(Bitmap src, Rectangle r)
        {
            Bitmap dst = new Bitmap(r.Width, r.Height);
            using (Graphics g = Graphics.FromImage(dst))
            {
                g.DrawImage(src,
                    new Rectangle(0, 0, r.Width, r.Height),
                    r,
                    GraphicsUnit.Pixel);
            }
            return dst;
        }

        public static Bitmap ResizeToSquare(Bitmap src, int size)
        {
            Bitmap dst = new Bitmap(size, size);
            using (Graphics g = Graphics.FromImage(dst))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBilinear;
                g.SmoothingMode = SmoothingMode.None;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.DrawImage(src, 0, 0, size, size);
            }
            return dst;
        }

        // Преобразование бинарной картинки size×size в вектор size*size (0/1)
        public static double[] ToInputVector(Bitmap binarySquare)
        {
            int w = binarySquare.Width;
            int h = binarySquare.Height;
            double[] v = new double[w * h];

            int k = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    v[k++] = (binarySquare.GetPixel(x, y).R == 0) ? 1.0 : 0.0;
                }

            return v;
        }

        /// Полный пайплайн: кадр -> бинаризация -> crop -> resize -> вектор
        public static double[] PreprocessToVector(Bitmap frame, int size, byte threshold, int padding)
        {
            Bitmap bin = null;
            Bitmap cropped = null;
            Bitmap resized = null;

            try
            {
                bin = ToBinary(frame, threshold);
                Rectangle bounds = FindContentBounds(bin, padding);
                cropped = Crop(bin, bounds);
                resized = ResizeToSquare(cropped, size);
                return ToInputVector(resized);
            }
            finally
            {
                if (resized != null) resized.Dispose();
                if (cropped != null) cropped.Dispose();
                if (bin != null) bin.Dispose();
            }
        }

        /// Для отображения: кадр -> бинаризация -> crop -> resize (картинка)
        public static Bitmap PreprocessToBitmap(Bitmap frame, int size, byte threshold, int padding)
        {
            Bitmap bin = ToBinary(frame, threshold);
            Rectangle bounds = FindContentBounds(bin, padding);
            Bitmap cropped = Crop(bin, bounds);
            Bitmap resized = ResizeToSquare(cropped, size);

            // чистим промежуточное
            cropped.Dispose();
            bin.Dispose();

            return resized; // caller Dispose()
        }

        // Повернуть бинарное изображение вокруг центра на angleDeg (фон белый)
        private static Bitmap Rotate(Bitmap src, float angleDeg)
        {
            Bitmap dst = new Bitmap(src.Width, src.Height);
            using (Graphics g = Graphics.FromImage(dst))
            {
                g.Clear(Color.White);
                g.InterpolationMode = InterpolationMode.NearestNeighbor; // для бинарки лучше так
                g.TranslateTransform(src.Width / 2f, src.Height / 2f);
                g.RotateTransform(angleDeg);
                g.TranslateTransform(-src.Width / 2f, -src.Height / 2f);
                g.DrawImage(src, 0, 0);
            }
            return dst;
        }

        // Оценка угла наклона по PCA чёрных пикселей (очень рабочий "deskew")
        private static float EstimateSkewAngleDeg(Bitmap bin)
        {
            // Собираем координаты чёрных пикселей
            double sumX = 0, sumY = 0;
            int n = 0;

            for (int y = 0; y < bin.Height; y++)
                for (int x = 0; x < bin.Width; x++)
                    if (bin.GetPixel(x, y).R == 0)
                    {
                        sumX += x;
                        sumY += y;
                        n++;
                    }

            if (n < 20) return 0f; // мало данных — не трогаем

            double mx = sumX / n;
            double my = sumY / n;

            // ковариации
            double sxx = 0, syy = 0, sxy = 0;
            for (int y = 0; y < bin.Height; y++)
                for (int x = 0; x < bin.Width; x++)
                    if (bin.GetPixel(x, y).R == 0)
                    {
                        double dx = x - mx;
                        double dy = y - my;
                        sxx += dx * dx;
                        syy += dy * dy;
                        sxy += dx * dy;
                    }

            // угол главной оси: 0.5*atan2(2*sxy, sxx - syy)
            double theta = 0.5 * Math.Atan2(2 * sxy, (sxx - syy));
            float deg = (float)(theta * 180.0 / Math.PI);

            // Нормализуем: для твоих символов повороты обычно небольшие,
            // ограничим, чтобы не "переворачивало" знак.
            if (deg > 45) deg -= 90;
            if (deg < -45) deg += 90;

            return deg;
        }

        // Масштабирование с сохранением пропорций + паддинг до size×size
        private static Bitmap ResizeKeepAspectAndPad(Bitmap src, int size, int pad = 2)
        {
            int w = src.Width, h = src.Height;
            if (w <= 0 || h <= 0) return new Bitmap(size, size);

            // целевой размер под контент
            int target = size - 2 * pad;
            double scale = Math.Min(target / (double)w, target / (double)h);
            int nw = Math.Max(1, (int)Math.Round(w * scale));
            int nh = Math.Max(1, (int)Math.Round(h * scale));

            Bitmap dst = new Bitmap(size, size);
            using (Graphics g = Graphics.FromImage(dst))
            {
                g.Clear(Color.White);
                g.InterpolationMode = InterpolationMode.NearestNeighbor;
                int ox = (size - nw) / 2;
                int oy = (size - nh) / 2;
                g.DrawImage(src, new Rectangle(ox, oy, nw, nh));
            }
            return dst;
        }

        // Сдвиг так, чтобы центр масс чёрных пикселей попал в центр
        private static Bitmap CenterByMass(Bitmap binSquare)
        {
            double sumX = 0, sumY = 0;
            int n = 0;

            for (int y = 0; y < binSquare.Height; y++)
                for (int x = 0; x < binSquare.Width; x++)
                    if (binSquare.GetPixel(x, y).R == 0)
                    {
                        sumX += x;
                        sumY += y;
                        n++;
                    }

            if (n < 20) return (Bitmap)binSquare.Clone();

            double cx = sumX / n;
            double cy = sumY / n;

            int tx = (int)Math.Round(binSquare.Width / 2.0 - cx);
            int ty = (int)Math.Round(binSquare.Height / 2.0 - cy);

            Bitmap dst = new Bitmap(binSquare.Width, binSquare.Height);
            using (Graphics g = Graphics.FromImage(dst))
            {
                g.Clear(Color.White);
                g.InterpolationMode = InterpolationMode.NearestNeighbor;
                g.DrawImage(binSquare, tx, ty);
            }
            return dst;
        }

        // Новый пайплайн: bin -> crop -> deskew -> resize keep aspect -> center -> vector
        public static double[] PreprocessToVectorSmart(Bitmap frame, int size, byte threshold, int padding)
        {
            Bitmap bin = null;
            Bitmap cropped = null;
            Bitmap rotated = null;
            Bitmap resized = null;
            Bitmap centered = null;

            try
            {
                bin = ToBinary(frame, threshold);

                Rectangle bounds = FindContentBounds(bin, padding);
                cropped = Crop(bin, bounds);

                float angle = EstimateSkewAngleDeg(cropped);
                rotated = Rotate(cropped, -angle); // поворачиваем "назад", убирая наклон

                // после поворота может появиться новый белый фон — снова обрежем по контенту
                Rectangle b2 = FindContentBounds(rotated, padding);
                using (var cropped2 = Crop(rotated, b2))
                {
                    resized = ResizeKeepAspectAndPad(cropped2, size, pad: 2);
                }

                centered = CenterByMass(resized);

                return ToInputVector(centered);
            }
            finally
            {
                centered?.Dispose();
                resized?.Dispose();
                rotated?.Dispose();
                cropped?.Dispose();
                bin?.Dispose();
            }
        }

        public static Bitmap PreprocessToBitmapSmart(Bitmap frame, int size, byte threshold, int padding)
        {
            Bitmap bin = null;
            Bitmap cropped = null;
            Bitmap rotated = null;
            Bitmap resized = null;
            Bitmap centered = null;

            try
            {
                bin = ToBinary(frame, threshold);
                Rectangle bounds = FindContentBounds(bin, padding);
                cropped = Crop(bin, bounds);

                float angle = EstimateSkewAngleDeg(cropped);
                rotated = Rotate(cropped, -angle);

                Rectangle b2 = FindContentBounds(rotated, padding);
                using (var cropped2 = Crop(rotated, b2))
                    resized = ResizeKeepAspectAndPad(cropped2, size, pad: 2);

                centered = CenterByMass(resized);

                return (Bitmap)centered.Clone();
            }
            finally
            {
                centered?.Dispose();
                resized?.Dispose();
                rotated?.Dispose();
                cropped?.Dispose();
                bin?.Dispose();
            }
        }
    }
}
