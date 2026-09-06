using System.Text.RegularExpressions;

namespace PostThinkRP
{
    public static class TagSanitizer
    {
        private static readonly Regex KnownBlocks = new Regex(
            "<think>.*?</think>|<post-thinking>.*?</post-thinking>",
            RegexOptions.Singleline | RegexOptions.IgnoreCase);

        private static readonly Regex GenericTags = new Regex(
            "</?[^>\\n]{1,80}>");

        public static string StripHiddenTags(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                return string.Empty;
            }

            var clean = KnownBlocks.Replace(text, string.Empty);
            var postIndex = clean.ToLowerInvariant().IndexOf("<post-thinking>");
            if (postIndex >= 0)
            {
                clean = clean.Substring(0, postIndex);
            }

            var thinkIndex = clean.ToLowerInvariant().IndexOf("<think>");
            if (thinkIndex >= 0)
            {
                clean = clean.Substring(0, thinkIndex);
            }

            clean = clean.Replace("</post-thinking>", string.Empty)
                         .Replace("</think>", string.Empty);
            clean = GenericTags.Replace(clean, string.Empty);
            return Regex.Replace(clean, "\\s{2,}", " ").Trim();
        }
    }
}
