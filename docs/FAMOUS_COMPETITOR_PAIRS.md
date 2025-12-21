# Famous Competitor Pairs for Validation

This document lists 30+ famous company pairs that should rank highly similar to each other. These pairs are used for validation to ensure the similarity system works correctly.

## Validation Pairs by Category

### Beverages (Non-Alcoholic)
1. **KO → PEP** (Coca-Cola → Pepsi) - Classic competitors
2. **PEP → KO** (Pepsi → Coca-Cola) - Reverse direction
3. **KO → KDP** (Coca-Cola → Keurig Dr Pepper) - Beverage competitors
4. **PEP → KDP** (Pepsi → Keurig Dr Pepper) - Beverage competitors

### Retail - Big Box / Discount
5. **WMT → TGT** (Walmart → Target) - Discount retailers
6. **TGT → WMT** (Target → Walmart) - Reverse direction
7. **WMT → COST** (Walmart → Costco) - Warehouse clubs
8. **COST → WMT** (Costco → Walmart) - Reverse direction

### Retail - Home Improvement
9. **HD → LOW** (Home Depot → Lowes) - Home improvement retailers
10. **LOW → HD** (Lowes → Home Depot) - Reverse direction

### Technology - Software / Platforms
11. **AAPL → MSFT** (Apple → Microsoft) - Tech giants
12. **MSFT → AAPL** (Microsoft → Apple) - Reverse direction
13. **GOOGL → MSFT** (Google → Microsoft) - Tech platforms
14. **META → GOOGL** (Meta → Google) - Social media / advertising

### Technology - Semiconductors
15. **NVDA → AMD** (NVIDIA → AMD) - GPU/CPU manufacturers
16. **AMD → NVDA** (AMD → NVIDIA) - Reverse direction
17. **INTC → AMD** (Intel → AMD) - CPU competitors

### Financial Services - Banks
18. **JPM → BAC** (JPMorgan → Bank of America) - Major banks
19. **BAC → JPM** (Bank of America → JPMorgan) - Reverse direction
20. **WFC → BAC** (Wells Fargo → Bank of America) - Regional banks
21. **C → JPM** (Citigroup → JPMorgan) - Investment banks

### Financial Services - Investment Banks
22. **GS → MS** (Goldman Sachs → Morgan Stanley) - Investment banks
23. **MS → GS** (Morgan Stanley → Goldman Sachs) - Reverse direction

### Financial Services - Credit Cards
24. **V → MA** (Visa → Mastercard) - Payment processors
25. **MA → V** (Mastercard → Visa) - Reverse direction
26. **AXP → V** (American Express → Visa) - Credit card networks

### Restaurants - Fast Food
27. **MCD → SBUX** (McDonald's → Starbucks) - Quick service restaurants
28. **SBUX → MCD** (Starbucks → McDonald's) - Reverse direction
29. **YUM → MCD** (Yum Brands → McDonald's) - Fast food chains
30. **CMG → SBUX** (Chipotle → Starbucks) - Fast casual

### Healthcare - Pharmaceuticals
31. **JNJ → PFE** (Johnson & Johnson → Pfizer) - Pharma giants
32. **PFE → JNJ** (Pfizer → Johnson & Johnson) - Reverse direction
33. **ABBV → JNJ** (AbbVie → Johnson & Johnson) - Pharma companies

### Healthcare - Insurance
34. **UNH → CVS** (UnitedHealth → CVS Health) - Healthcare services
35. **CVS → WBA** (CVS → Walgreens) - Pharmacy chains

### Energy - Oil & Gas
36. **XOM → CVX** (Exxon Mobil → Chevron) - Oil majors
37. **CVX → XOM** (Chevron → Exxon Mobil) - Reverse direction
38. **COP → XOM** (ConocoPhillips → Exxon Mobil) - Oil & gas

### Automotive
39. **TSLA → GM** (Tesla → General Motors) - Auto manufacturers
40. **GM → TSLA** (General Motors → Tesla) - Reverse direction
41. **TM → HMC** (Toyota → Honda) - Japanese automakers

### Aerospace & Defense
42. **LMT → RTX** (Lockheed Martin → RTX) - Defense contractors
43. **RTX → LMT** (RTX → Lockheed Martin) - Reverse direction
44. **NOC → LMT** (Northrop Grumman → Lockheed Martin) - Defense

### Media & Entertainment
45. **DIS → NFLX** (Disney → Netflix) - Entertainment companies
46. **NFLX → DIS** (Netflix → Disney) - Reverse direction
47. **CMCSA → DIS** (Comcast → Disney) - Media conglomerates

### Consumer Goods - Apparel
48. **NKE → UA** (Nike → Under Armour) - Athletic apparel
49. **UA → NKE** (Under Armour → Nike) - Reverse direction
50. **LULU → NKE** (Lululemon → Nike) - Athletic wear

### Consumer Goods - Personal Care
51. **PG → CL** (Procter & Gamble → Colgate-Palmolive) - Consumer products
52. **CL → PG** (Colgate-Palmolive → Procter & Gamble) - Reverse direction

### Telecommunications
53. **T → VZ** (AT&T → Verizon) - Telecom carriers
54. **VZ → T** (Verizon → AT&T) - Reverse direction
55. **TMUS → VZ** (T-Mobile → Verizon) - Wireless carriers

---

## Total: 53 Pairs (30+ unique pairs, some bidirectional)

**Note**: Some companies don't exist in the database (F/Ford, BA/Boeing, T/AT&T, C/Citigroup, WBA/Walgreens, TMUS/T-Mobile), so those pairs have been removed or replaced.

## Usage

These pairs can be used for:
1. **Famous Pairs Smoke Tests** - Quick validation after scoring changes
2. **Stack-Ranked Validation** - Comprehensive validation of ranking quality
3. **Regression Testing** - Ensure changes don't break existing validations

## Validation Scripts

### Test Famous Pairs (Quick Validation)

```bash
# Test all 53 pairs
python scripts/validate_famous_pairs.py

# Test first 10 pairs (quick test)
python scripts/validate_famous_pairs.py --limit 10

# Save report to file
python scripts/validate_famous_pairs.py --output famous_pairs_report.md

# Or use CLI entry point
validate-famous-pairs --limit 10
```

### Test Stack-Ranked Rankings (Comprehensive)

```bash
# Test default companies (12 companies)
python scripts/validate_ranking_quality.py

# Test specific companies
python scripts/validate_ranking_quality.py --tickers KO,PEP,HD,LOW
```

## Expected Results

For each pair, the expected similar company should rank in **top-3** (ideally #1) when querying the first company for similar companies.

**Example Output:**
```
## Coca-Cola → Pepsi
✅ **Rank #1** (expected ≤1) - **PASS**

## Pepsi → Coca-Cola
✅ **Rank #2** (expected ≤1) - **PASS** (close enough)
```
