# Security Policy

## Supported versions

安全修复仅应用于默认分支的最新版本。

## Reporting

请使用 GitHub 的 **Private vulnerability reporting** 私下提交漏洞，不要在公开 Issue
中披露可利用细节。维护者会尽量在 7 天内确认，并在完成修复和协调披露后发布公告。

## Scope

请勿提交验证码绕过服务、真实账户攻击或未经授权的数据。模型文件来自外部 Release 时，
使用前应核对发布页校验值，并仅加载可信来源的 PyTorch checkpoint。
