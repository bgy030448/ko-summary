import unittest
from app import preprocess_news


class NewsPreprocessTests(unittest.TestCase):
    def test_pasted_news(self):
        raw = '''전지윤(36)이 11월의 신부가 된다.
[깜짝 결혼](https://img-s-msn-com.akamaized.net/tenant/amp/entityid/test.img?w=534\\&h=713)
깜짝 결혼
전지윤이 방송국에 들어서며 포즈를 취하고 있다. / 뉴스1
오는 22일 비연예인과 결혼한다.
[**엑셀 워드 PPT까지 최대 100개 관리 가능해요.**](https://www.bing.com/api/v2/mediation/tracking?x=1)
**Adobe Korea**[후원받음](https://www.bing.com/api/v2/mediation/tracking?x=2)
[call to action icon](https://www.bing.com/api/v2/mediation/tracking?x=3 "광고") ([image](https://assets.msn.com/icons/test.svg))
1990년생으로 2009년 데뷔했다.'''
        self.assertEqual(preprocess_news(raw),
                         '전지윤(36)이 11월의 신부가 된다. 오는 22일 비연예인과 결혼한다. 1990년생으로 2009년 데뷔했다.')

    def test_preserve_article_facts_and_link_labels(self):
        raw = '광고 시장은 10.5% 성장했다.\n[보고서](https://example.com/report_(2026))에 따르면 그렇다.\n광고\n후원받음'
        self.assertEqual(preprocess_news(raw),
                         '광고 시장은 10.5% 성장했다. 보고서에 따르면 그렇다.')

    def test_idempotent(self):
        once = preprocess_news('첫 문장이다.\n\n두 번째 문장이다.\u200b')
        self.assertEqual(preprocess_news(once), once)


if __name__ == '__main__':
    unittest.main()
